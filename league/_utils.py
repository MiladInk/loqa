import math
import os
import zipfile
from typing import Sequence, Any

import jax
import numpy as np
from flax import linen as nn, struct
from jax import numpy as jp, random as rax

eps = 1e-8

def slurm_infos():
    return {
        'slurm/job_id': os.getenv('SLURM_JOB_ID'),
        'slurm/job_user': os.getenv('SLURM_JOB_USER'),
        'slurm/job_partition': os.getenv('SLURM_JOB_PARTITION'),
        'slurm/cpus_per_node': os.getenv('SLURM_JOB_CPUS_PER_NODE'),
        'slurm/num_nodes': os.getenv('SLURM_JOB_NUM_NODES'),
        'slurm/nodelist': os.getenv('SLURM_JOB_NODELIST'),
        'slurm/cluster_name': os.getenv('SLURM_CLUSTER_NAME'),
        'slurm/array_task_id': os.getenv('SLURM_ARRAY_TASK_ID')
    }


class FILM(nn.Module):
  size: int
  @nn.compact
  def __call__(self, x, c):
    """Applies layer normalization on the input.

    Args:
      x: the inputs
      c: context that we want to condition on, here it is the reward sharing matrix
    Returns:
      inputs mutiplied and biased conditioned on "c" attribute
    """
    features = x.shape[-1]

    scale = nn.Dense(features=self.size)(c)
    scale = nn.relu(scale)
    scale = nn.Dense(features=features)(scale)

    bias = nn.Dense(features=self.size)(c)
    bias = nn.relu(bias)
    bias = nn.Dense(features=features)(bias)

    y = x*scale+bias
    return y


def global_norm(updates):
    # taken from https://github.com/deepmind/optax/blob/9dbf9366996c4daeaf0bdc8394aa3f79a7946949/optax/_src/clipping.py
    return jp.sqrt(sum(jp.sum(x ** 2) for x in jax.tree_util.tree_leaves(updates)))

def grad_max(grads):
    if grads.params is None:
        return 0.0
    return jp.max(jp.concatenate(jax.tree_leaves(jax.tree_map(lambda x: jp.abs(x).max().reshape(1), grads))))

def clip_grads(grads, max_grad):
    return jax.tree_map(lambda dx: jp.clip(dx, -max_grad, +max_grad), grads)

def clip_by_l2_norm(x, max_norm):
    # taken from https://github.com/deepmind/rlax/blob/dc048fdbc8903dd001ccd899f45d744bb3c1e2c6/rlax/_src/policy_gradients.py
    sum_sq = jp.sum(jp.vdot(x, x))
    nonzero = sum_sq > 0
    sum_sq_ones = jp.where(nonzero, sum_sq, jp.ones_like(sum_sq))
    norm = jp.where(nonzero, jp.sqrt(sum_sq_ones), sum_sq)
    return (x * max_norm) / jp.maximum(norm, max_norm)

def clip_grads_by_norm(updates, max_norm):
    # taken from https://github.com/deepmind/optax/blob/9dbf9366996c4daeaf0bdc8394aa3f79a7946949/optax/_src/clipping.py
    g_norm = global_norm(updates)
    trigger = jp.squeeze(g_norm < max_norm)

    def clip_fn(t):
        return jax.lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * max_norm)

    updates = jax.tree_util.tree_map(clip_fn, updates)
    return updates

def warmup_then_fixed_lr_schedule(target_lr: float, warmup_steps: int):
    def schedule(step):
        return jp.where(step < warmup_steps, target_lr * (step / warmup_steps), target_lr)
    return schedule

class PositionalEncoding(nn.Module):
    d_model: int         # Hidden dimensionality of the input.
    max_len: int = 128  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = self.param(name='pe', init_fn=lambda *_: pe)

    def __call__(self, x):
        """
        x: [SeqLen, HiddenDim]
        returns: [SeqLen * 2, HiddenDim]
        """
        assert len(x.shape) == 2, f'Expected 2D input, got {x.shape}'
        # assert x.shape[1] == self.d_model, f'Expected input with {self.d_model} features, got {x.shape[1]}'

        x = jp.concatenate([x, self.pe[:x.shape[0]]], axis=1)
        return x


class MLP(nn.Module):
  features: Sequence[int]
  last_activation: bool = False

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    if self.last_activation:
        x = nn.relu(x)
    return x

class MLPResidual(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    assert len(self.features) >= 2
    x = nn.Dense(self.features[0])(x)
    for feat in self.features[1:]:
      x = x + nn.relu(nn.Dense(feat)(x))
    return x

class MLPResidualLayerNorm(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
      assert len(self.features) >= 2
      x = nn.Dense(self.features[0])(x)
      for feat in self.features[1:]:
          x = nn.LayerNorm()(x)
          x = x + nn.relu(nn.Dense(feat)(x))
      return x


def cross_entropy(logits1, logits2):
    p1 = jax.nn.softmax(logits1)
    logp1 = jax.nn.log_softmax(logits1)
    logp2 = jax.nn.log_softmax(logits2)
    return (p1 * (logp1 - logp2)).sum()

def policy_gradient_loss(logps, advantages):
    advantages = jax.lax.stop_gradient(advantages)  # [trace_length]
    return jp.einsum('t,t->', logps, advantages) * -1  # [1]

def policy_loss(logps, values, rewards, hp):
    assert logps.shape[0] == values.shape[0] - 1 == rewards.shape[0]
    if hp['advantage_estimation_mode'] == 'td0':
        advantages = rewards + hp['reward_discount'] * values[1:] - values[:-1]
    elif hp['advantage_estimation_mode'] == 'monte_carlo_no_discount_yes_baseline':
        advantages = revcumsum(rewards, axis=0) - values[:-1]
    elif hp['advantage_estimation_mode'] == 'monte_carlo_yes_discount_yes_baseline':
        advantages = discounted_returns(rewards, hp['reward_discount']) - values[:-1]
    elif hp['advantage_estimation_mode'] == 'gae':
        advantages = gae(rewards, values, discount_factor=hp['reward_discount'], gamma=hp['gae_lambda'])
    else:
        raise f"Unknown mode {hp['advantage_estimation_mode']}"
    return policy_gradient_loss(logps, advantages)

def value_loss(rewards, target_values, values, hp):
    if hp['value_algorithm_mode'] == 'td0':
        targets = target_values[1:] * hp['reward_discount'] + rewards
    elif hp['value_algorithm_mode'] == 'monte-carlo':
        targets = discounted_returns(rewards, hp['reward_discount'])
    else:
        raise f"Unknown value algorithm mode: {hp['value_algorithm_mode']}"

    td_error = values[:-1] - jax.lax.stop_gradient(targets)

    if hp['final_state_value_zero']:
        td_error = jp.concatenate([td_error, values[-1][None] - 0.], axis=0)  # the target value for the final state is 0.
        assert td_error.shape == (hp['trace_length'] + 1,)
    else:
        assert td_error.shape == (hp['trace_length'],)

    if hp['value_loss_mode'] == 'mse':
        return (td_error ** 2).mean()
    elif hp['value_loss_mode'] == 'huber':
        return huber_loss(td_error).mean()
    else:
        raise f"Unknown value loss mode: {hp['value_loss_mode']}"


def revcumsum(x, axis):
  return jp.flip(jp.cumsum(jp.flip(x, axis=axis), axis=axis), axis=axis)


def huber_loss(x, delta: float = 1.):
    # taken from https://github.com/deepmind/rlax/blob/f1ad41f79d617551911da4fd61acca99d8fea84c/rlax/_src/clipping.py
    # 0.5 * x^2                  if |x| <= d
    # 0.5 * d^2 + d * (|x| - d)  if |x| > d
    abs_x = jp.abs(x)
    quadratic = jp.minimum(abs_x, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_x - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


@struct.dataclass
class Optimizer:
    opt: Any = struct.field(pytree_node=False)
    opt_state: Any


def get_lr(hp, lr):
    if hp['lr_schedule'] == 'constant':
        def schedule(_):
            return lr
        return schedule
    elif hp['lr_schedule'] == 'warmup_constant':
        return warmup_then_fixed_lr_schedule(target_lr=lr, warmup_steps=hp['lr_warmup_steps'])
    else:
        raise f"Unknown lr_schedule: {hp['lr_schedule']}"


def update_target(old_target, new, value_target_update_config):
    target_update_config=value_target_update_config
    if target_update_config['mode'] == 'ema':
        tau = target_update_config['tau']
        old_target_params = old_target.params
        new_params = new.params
        new_target_params = jax.tree_map(lambda old, new: old * tau + new * (1 - tau), old_target_params, new_params)
        return old_target.replace(params=new_target_params)
    else:
        raise f'Unknown target_update_config: {target_update_config}'


@jax.jit
def add_noise(tree, noise_scale, seed, mask):
    leaves, tree_def = jax.tree_util.tree_flatten(tree)
    seeds = rax.split(seed, len(leaves))
    seed_tree = jax.tree_util.tree_unflatten(tree_def, seeds)
    noisy_tree = jax.tree_map(lambda x, s, m: x + m*rax.normal(key=s, shape=x.shape, dtype=x.dtype) * noise_scale, tree, seed_tree, mask)
    return noisy_tree


def discounted_returns(rewards, discount_factor):
    def _body(c, r):
        a = r + discount_factor * c
        return a, a
    _, returns = jax.lax.scan(f=_body, init=0., xs=jp.flip(rewards))
    return jp.flip(returns)

def gae(rewards, values, discount_factor, gamma):
    assert values.shape[0] - 1 == rewards.shape[0]  # ex: values = [v1, v2, v3], rewards = [r1, r2] discount_factor = 0.9, gamma = 0.8
    td0s = rewards + discount_factor * values[1:] - values[:-1]  # ex: td0s = [r1 + 0.9 * v2 - v1, r2 + 0.9 * v3 - v2] = [d1, d2]

    def _body(c, td0):
        a = td0 + discount_factor * gamma * c
        return a, a

    _, aes = jax.lax.scan(f=_body, init=0., xs=jp.flip(td0s))
    advantage_estimates = jp.flip(aes)  # ex: td0s = [d1 + 0.9 * 0.8 * d2, d2]

    return advantage_estimates

def test_discounted_returns():
    ans = discounted_returns(jp.array([1., 2, 3], dtype=float), jp.array(0.92))
    print('ans', ans)
    assert jp.allclose(ans, jp.array([5.3792, 4.76, 3.], dtype=float)).all()



def zip_directory(directory, general_save_path):
    zip_filename = 'files.zip'
    zip_save_path = os.path.join(general_save_path, zip_filename)
    with zipfile.ZipFile(zip_save_path, 'w') as zip_file:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not (d.startswith("_") or 'wandb' in d)]
            for file in files:
                if file.endswith(".py") or file.endswith(".sh"):
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path)
    return zip_save_path

def discretize(x, bins):
  return jp.round(x * bins)/bins  # non-differentiable

def straight_through_discretize(x, bins):
  # an exactly-one gradient.
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(discretize(x, bins=bins))

def straight_through_integer(x):
    # an exactly-one gradient.
    zero = x - jax.lax.stop_gradient(x)
    return zero + jax.lax.stop_gradient(jp.round(x))

# taken from https://github.com/Silent-Zebra/POLA/blob/master/jax_files/POLA_dice_jax.py
# removed the linear_end as we have actor and value head use the output of the shared GRU
class POLAGruCell(nn.Module):
    num_outputs: int
    num_hidden_units: int
    layers_before_gru: int

    def setup(self):
        if self.layers_before_gru >= 1:
            self.linear1 = nn.Dense(features=self.num_hidden_units)
        if self.layers_before_gru >= 2:
            self.linear2 = nn.Dense(features=self.num_hidden_units)
        self.GRUCell = nn.GRUCell()
        # self.linear_end = nn.Dense(features=self.num_outputs)

    def __call__(self, carry_0, x):
        if self.layers_before_gru >= 1:
            x = self.linear1(x)
            x = nn.relu(x)
        if self.layers_before_gru >= 2:
            x = self.linear2(x)

        carry, x = self.GRUCell(carry_0, x)
        # outputs = self.linear_end(x)
        return carry, ({'h': x, 'c0': carry_0})

class POLAGRU(nn.Module):
    num_outputs: int
    context_size: int
    layers_before_gru: int

    @nn.compact
    def __call__(self, x, carry=None):
        gru = nn.scan(POLAGruCell,
                     variable_broadcast="params",
                     split_rngs={"params": False},
                     in_axes=0,
                     out_axes=0)

        if carry is None:
            carry_0 = self.get_initial_carry()
        else:
            carry_0 = carry

        carry, outs = gru(num_outputs=self.num_outputs,
                          num_hidden_units=self.context_size,
                          layers_before_gru=self.layers_before_gru)(carry_0, x)

        hs = outs['h']
        carries_0 = outs['c0']

        return {'hs': hs, 'carry': carry, 'carries_0': carries_0}

    def get_initial_carry(self):
        return nn.initializers.zeros(key=rax.PRNGKey(42), shape=(self.context_size,))

def rlax_entropy_loss(logits_t, w_t=None):
    ps = jax.nn.softmax(logits_t)
    logps = jax.nn.log_softmax(logits_t)
    return (ps * logps).sum(axis=-1).mean()


def rscope(rng, *path):
  if rng.ndim > 1:  # deal with leading batch axes
    return jax.vmap(lambda rng: rscope(rng, *path))(rng)
  # NOTE used to use seed = hash(path) but this is nondeterministic
  import zlib
  data = "/".join(path).encode("ascii")
  seed = zlib.crc32(data)
  return rax.fold_in(rng, seed)


def magic_box(z):
    return jp.exp(z - jax.lax.stop_gradient(z))

def npify(tree):
    return jax.tree_map(lambda p: np.array(p), tree)
