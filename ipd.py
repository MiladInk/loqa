import collections
import os
import random
import shutil
import time
from collections import Counter
from functools import partial
from typing import Any, Sequence

import flax
import jax
import numpy as np
import optax
from jax import config, random as rax, numpy as jp
from flax import struct
import flax.linen as nn
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from utils import slurm_infos, AliasDict, rscope, clip_grads_by_norm

rng_dummy = rax.PRNGKey(0)

config.update('jax_disable_jit', False)


def magic_box(z):
    return jp.exp(z - jax.lax.stop_gradient(z))


def ipd_step(actions):
    """
    actions: 0=cooperate, 1:defect, shape: [2x1] = [first player action, second player action], dtype=int
    """
    state = actions[0] * 2 + actions[1]  # CC=0, CD=1, DC=2, DD=3

    # calculate rewards
    player1_rewards = jp.array([-1., -3., 0, -2.])
    player2_rewards = jp.array([-1., 0., -3, -2.])
    r1 = player1_rewards[state]
    r2 = player2_rewards[state]
    rewards = jp.stack([r1, r2])

    # calculate observation shape: [4x1] = [player1_cooperated, player1_defected, player2_cooperated, player2_defected]
    observation_agent0 = jp.stack([actions[0] == 0, actions[0] == 1, actions[1] == 0, actions[1] == 1])
    observation_agent1 = jp.stack([actions[1] == 0, actions[1] == 1, actions[0] == 0, actions[0] == 1])
    observation = jp.stack([observation_agent0, observation_agent1], axis=0)
    observation = observation.astype(jp.float32)

    return observation, rewards


class MLP(nn.Module):
    features: Sequence[int]
    activation_str: str

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat, name=f'dense_{i}')(x)
            if self.activation_str == 'relu':
                x = nn.relu(x)
            elif self.activation_str == 'tanh':
                x = nn.tanh(x)
            else:
                raise ValueError(f'activation {self.activation_str} not recognized')

        x = nn.Dense(self.features[-1], name=f'dense_{len(self.features) - 1}')(x)
        return x


class IPDLogits(nn.Module):
    @nn.compact
    def __call__(self, obs):
        ls_start = self.param('start', nn.initializers.zeros, (2,))
        ls_cc = self.param('CC', nn.initializers.zeros, (2,))
        ls_cd = self.param('CD', nn.initializers.zeros, (2,))
        ls_dc = self.param('DC', nn.initializers.zeros, (2,))
        ls_dd = self.param('DD', nn.initializers.zeros, (2,))

        out = jp.ones(2) * -1
        out = jp.where(jp.allclose(obs, jp.array([0., 0, 0, 0])), ls_start, out)
        out = jp.where(jp.allclose(obs, jp.array([1., 0, 1, 0])), ls_cc, out)
        out = jp.where(jp.allclose(obs, jp.array([1., 0, 0, 1])), ls_cd, out)
        out = jp.where(jp.allclose(obs, jp.array([0., 1, 1, 0])), ls_dc, out)
        out = jp.where(jp.allclose(obs, jp.array([0., 1, 0, 1])), ls_dd, out)
        return out


# taken from https://github.com/Silent-Zebra/POLA/blob/master/jax_files/POLA_dice_jax.py
class POLAGruCell(nn.Module):
    num_outputs: int
    num_hidden_units: int
    layers_before_gru: int

    def setup(self):
        if self.layers_before_gru >= 1:
            self.linear1 = nn.Dense(features=self.num_hidden_units)
        if self.layers_before_gru >= 2:
            self.linear2 = nn.Dense(features=self.num_hidden_units)
        self.GRUCell = nn.GRUCell(self.num_hidden_units)
        self.linear_end = nn.Dense(features=self.num_outputs)

    def __call__(self, carry_0, x):
        if self.layers_before_gru >= 1:
            x = self.linear1(x)
            x = nn.relu(x)
        if self.layers_before_gru >= 2:
            x = self.linear2(x)

        carry, x = self.GRUCell(carry_0, x)
        x = self.linear_end(x)
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


class GRUIPDAgent(nn.Module):

    def setup(self):
        self.actor_head = POLAGRU(2, 32, 2)
        self.qvalue_head = POLAGRU(2, 32, 2)

    def __call__(self, x):  # just used for initialization
        obs_seq = x['obs_seq']
        rng = x['rng']
        t = x['t']
        self.call_seq({'obs_seq': obs_seq})
        self.call_step({'carry_actor': self.actor_head.get_initial_carry(),
                        'carry_qvalue': self.qvalue_head.get_initial_carry(),
                        'obs': obs_seq[t],
                        'rng': rng,
                        't': t})

    def get_initial_carries(self):
        return {'carry_actor': self.actor_head.get_initial_carry(),
                'carry_qvalue': self.qvalue_head.get_initial_carry()}

    def call_seq(self, x):
        logp_seq = self.logp_seq(x)['logp_seq']  # (T, 2)
        qvalue_seq = self.qvalue_seq(x)['qvalue_seq']  # (T, 2)
        value_seq = (jp.exp(logp_seq) * qvalue_seq).sum(axis=-1)  # (T,)
        return {'logp_seq': logp_seq, 'qvalue_seq': qvalue_seq, 'value_seq': value_seq}

    def logp_seq(self, x):
        obs_seq = x['obs_seq']  # (T, 4)
        logits_seq = self.actor_head(obs_seq, carry=None)['hs']  # (T, 2)
        logp_seq = nn.log_softmax(logits_seq, axis=-1)  # (T, 2)
        return {'logp_seq': logp_seq}

    def qvalue_seq(self, x):
        obs_seq = x['obs_seq']  # (T, 4)
        t_seq = jp.arange(obs_seq.shape[0])  # (T,)
        t_seq = t_seq.reshape(-1, 1)  # (T, 1)
        h_seq = jp.concatenate([obs_seq, t_seq], axis=-1)  # (T, 5)
        qvalue_seq = self.qvalue_head(h_seq, carry=None)['hs']  # (T, 2)
        return {'qvalue_seq': qvalue_seq}

    def call_step(self, x):
        obs = x['obs']  # (T, 4)
        rng = x['rng']
        t = x['t']
        carry_actor = x['carry_actor']
        carry_qvalue = x['carry_qvalue']
        out_actor = self.logp_step({'obs': obs, 'carry': carry_actor})  # (2,)
        logp = out_actor['logp']  # (2,)
        next_carry_actor = out_actor['carry']  # (7,)
        action = rax.categorical(rng, logp)  # (1,)
        out_qvalue = self.qvalue_step({'t': t, 'obs': obs, 'carry': carry_qvalue})  # (2,)
        qvalue = out_qvalue['qvalue']  # (2,)
        next_carry_qvalue = out_qvalue['carry']  # (7,)
        return {'logp': logp,
                'qvalue': qvalue,
                'carry_actor': next_carry_actor,
                'carry_qvalue': next_carry_qvalue,
                'action': action}

    def logp_step(self, x):
        obs = x['obs']  # (4,)
        carry = x['carry']
        actor_res = self.actor_head(x=obs[None, :], carry=carry)
        logits = actor_res['hs'][0]  # (2,)
        logp = nn.log_softmax(logits, axis=-1)  # (2,)
        next_carry = actor_res['carry']
        return {'logp': logp, 'carry': next_carry}

    def qvalue_step(self, x):
        obs = x['obs']  # (4,)
        carry = x['carry']
        t = x['t']
        h = jp.concatenate([obs, jp.array([t])], axis=-1)  # (5,)
        qvalue_res = self.qvalue_head(x=h[None, :], carry=carry)  # (2,)
        qvalue = qvalue_res['hs'][0]  # (2,)
        next_carry = qvalue_res['carry']
        return {'qvalue': qvalue, 'carry': next_carry}


class OneStepIPDAgentDense(nn.Module):
    actor_config: dict = struct.field(pytree_node=False)

    def setup(self):
        if self.actor_config['model'] == 'mlp':
            self.actor_head = MLP([8, 8, 2], self.actor_config['activation'])
        elif self.actor_config['model'] == 'logits':
            self.actor_head = IPDLogits()
        else:
            raise ValueError(f'actor model {self.actor_config["model"]} not recognized')
        self.qvalue_head = MLP([16, 16, 2], 'relu')

    def call_seq(self, x):
        obs_seq = x['obs_seq']
        times = jp.arange(obs_seq.shape[0])
        logp_seq = jax.vmap(lambda obs, t: self.logp({'obs': obs, 't': t})['logp'])(obs_seq, times)  # [T, 2]
        qvalue_seq = jax.vmap(lambda obs, t: self.qvalue({'obs': obs, 't': t})['qvalue'])(obs_seq, times)  # [T]
        value_seq = (jp.exp(logp_seq) * qvalue_seq).sum(axis=-1)  # (T,)
        return {'logp_seq': logp_seq, 'qvalue_seq': qvalue_seq, 'value_seq': value_seq}

    def __call__(self, x):  # flax needs this to use for initialization afaik
        obs = x['obs']
        rng = x['rng']
        t = x['t']

        res = self.logp({'obs': obs, 't': t})
        logits = res['logits']
        logp = res['logp']
        qvalue = self.qvalue({'obs': obs, 't': t})['qvalue']
        value = (jp.exp(logp) * qvalue).sum()

        if rng is not None:
            action = self.emit({'logp': logp, 'rng': rng})['action']
        else:
            action = None

        return {'logits': logits,
                'logp': logp,
                'qvalue': qvalue,
                'obs': obs,
                'h': obs,
                'rng': rng,
                'action': action,
                'value': value}

    def logp(self, x):
        h = x['obs']
        logits = self.actor_head(h)
        logp = jax.nn.log_softmax(logits, axis=-1)
        return {'logits': logits, 'logp': logp}

    def qvalue(self, x):
        obs = x['obs']
        t = x['t']
        h = jp.concatenate([obs, jp.array([t])])
        qvalue = self.qvalue_head(h)
        return {'qvalue': qvalue}

    def emit(self, x):
        logp = x['logp']
        rng = x['rng']
        return {'action': rax.categorical(rng, logp)}

    def logp_of_action(self, x):
        action = x['action']
        logp = self({'obs': x['obs']})['logp']
        return {'logp': logp[action]}


# noinspection PyUnresolvedReferences

@struct.dataclass
class IPDAgent:
    params: Any
    model: Any = struct.field(pytree_node=False)

    def __call__(self, *args, **kwargs):
        return self.model.apply(self.params, *args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.model, name):
            method = getattr(self.model, name)

            def method_wrapper(*args, **kwargs):
                return self.model.apply(self.params, *args, **kwargs, method=method)

            return method_wrapper

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


@struct.dataclass
class Optimizer:
    opt: Any = struct.field(pytree_node=False)
    opt_state: Any


@partial(jax.jit, static_argnames='hp')
def play_episode_scan(inp, hp):
    agent0 = inp['agent0']
    agent1 = inp['agent1']
    rng = inp['rng']

    def fn(carry, x):
        prev_obs = carry['prev_obs']
        r = x['rng']
        r, rng0, rng1 = rax.split(r, 3)
        out1 = agent0({'obs': prev_obs[0], 'rng': rng0, 't': 0})
        out2 = agent1({'obs': prev_obs[1], 'rng': rng1, 't': 0})
        r, rng2, rng3, rng4, rng5 = rax.split(r, 5)

        action1 = out1['action']
        random_action_1 = rax.randint(rng2, shape=[], minval=0, maxval=2)
        action1 = jp.where(rax.bernoulli(rng3, 1-hp['epsilon_greedy']), action1, random_action_1)

        action2 = out2['action']
        random_action_2 = rax.randint(rng4, shape=[], minval=0, maxval=2)
        action2 = jp.where(rax.bernoulli(rng5, 1-hp['epsilon_greedy']), action2, random_action_2)

        actions = jp.array([action1, action2])
        obs, rewards = ipd_step(actions)
        next_carry = {'prev_obs': obs}
        y = {'rew': rewards, 'act': actions, 'obs': obs}
        return next_carry, y

    rng, rng_episode = rax.split(rng, 2)
    carry_init = {'prev_obs': jp.zeros((2, 4))}
    xs = {'rng': rax.split(rng_episode, hp['game_length'])}
    _, ys = jax.lax.scan(fn, carry_init, xs, length=hp['game_length'])
    ys['obs'] = jp.concatenate([carry_init['prev_obs'][None, :], ys['obs']], axis=0)

    out = ys
    return out


@partial(jax.jit, static_argnames='hp')
def play_episode_scan_gru(inp, hp):
    assert hp['actor']['model'] == 'gru', 'model is not gru'

    agent0 = inp['agent0']
    agent1 = inp['agent1']
    rng = inp['rng']
    c_0_actor = inp['c_0_actor']
    c_0_qvalue = inp['c_0_qvalue']
    c_1_actor = inp['c_1_actor']
    c_1_qvalue = inp['c_1_qvalue']

    def fn(carry, x):
        prev_obs = carry['prev_obs']
        c_0_actor = carry['c_0_actor']
        c_1_actor = carry['c_1_actor']
        c_0_qvalue = carry['c_0_qvalue']
        c_1_qvalue = carry['c_1_qvalue']
        r = x['rng']
        t = x['t']
        rng0, rng1 = rax.split(r, 2)
        # The t is just used for qvalues which are thrown away in the next line.
        out1 = agent0.call_step({'obs': prev_obs[0], 'rng': rng0, 't': t, 'carry_actor': c_0_actor, 'carry_qvalue': c_0_qvalue})
        out2 = agent1.call_step({'obs': prev_obs[1], 'rng': rng1, 't': t, 'carry_actor': c_1_actor, 'carry_qvalue': c_1_qvalue})
        action1 = out1['action']
        action2 = out2['action']
        c_0_actor = out1['carry_actor']
        c_1_actor = out2['carry_actor']
        c_0_qvalue = out1['carry_qvalue']
        c_1_qvalue = out2['carry_qvalue']
        actions = jp.array([action1, action2])
        obs, rewards = ipd_step(actions)
        next_carry = {'prev_obs': obs, 'c_0_actor': c_0_actor, 'c_1_actor': c_1_actor, 'c_0_qvalue': c_0_qvalue, 'c_1_qvalue': c_1_qvalue}
        y = {'rew': rewards, 'act': actions, 'obs': obs}
        return next_carry, y

    rng, rng_episode = rax.split(rng, 2)
    carry_init = {'prev_obs': jp.zeros((2, 4)), 'c_0_actor': c_0_actor, 'c_1_actor': c_1_actor, 'c_0_qvalue': c_0_qvalue, 'c_1_qvalue': c_1_qvalue}
    xs = {'rng': rax.split(rng_episode, hp['game_length']), 't': jp.arange(hp['game_length'])}
    _, ys = jax.lax.scan(fn, carry_init, xs, length=hp['game_length'])
    ys['obs'] = jp.concatenate([carry_init['prev_obs'][None, :], ys['obs']], axis=0)

    out = ys
    return out


@partial(jax.jit, static_argnames='hp')
def gen_episodes(agent, opponents, rng, hp):
    batch_size = hp['batch_size']
    rng, *rng_batch = rax.split(rng, batch_size + 1)
    rngs = jp.stack(rng_batch)
    if hp['actor']['model'] in ['gru']:
        cs_0 = agent.get_initial_carries()
        cs_1 = opponents.get_initial_carries()
        carries = {'c_0_actor': cs_0['carry_actor'],
                   'c_1_actor': cs_1['carry_actor'],
                   'c_0_qvalue': cs_0['carry_qvalue'],
                   'c_1_qvalue': cs_1['carry_qvalue']}

        def func(r, op):
            return play_episode_scan_gru({'agent0': agent,
                                          'agent1': op,
                                          'rng': r,
                                          **carries},
                                         hp)

    elif hp['actor']['model'] in ['mlp', 'logits']:
        def func(r, op):
            return play_episode_scan({'agent0': agent,
                                      'agent1': op,
                                      'rng': r},
                                       hp)
    else:
        raise ValueError(f"Unknown model {hp['actor']['model']}")

    if use_rb(hp):
        episodes = jax.vmap(func)(rscope(rngs, "play_rng"), opponents)
    else:
        opponent = opponents[0]
        episodes = jax.vmap(lambda r: func(r,  opponent))(rscope(rngs, "play_rng"))

    return episodes


def global_norm(updates):
    # taken from https://github.com/deepmind/optax/blob/9dbf9366996c4daeaf0bdc8394aa3f79a7946949/optax/_src/clipping.py
    return jp.sqrt(sum(jp.sum(x ** 2) for x in jax.tree_util.tree_leaves(updates)))


@partial(jax.jit, static_argnames=('hp', 'include_opponent'))
def update_agent_actor(agent, opponents, optimizers, hp, episodes, player_to_train: int, include_opponent: bool = False):
    op_rewards_baseline = episodes['rew'][..., 1 - player_to_train].mean()

    def loss_fn(a):
        def func(ep, op):
            return agent_policy_loss(agent=a,
                                     opponent=op,
                                     hp=hp,
                                     episodes=ep,
                                     player_to_train=player_to_train,
                                     op_rewards_baseline=op_rewards_baseline,
                                     )

        if use_rb(hp):
            aux = jax.vmap(func, axis_name='batch')(episodes, opponents)
        else:
            opponent = opponents[0]
            aux = jax.vmap(lambda ep: func(ep, opponent), axis_name='batch')(episodes)

        loss_agent = aux['loss_agent'].mean() + hp['actor']['train']['entropy_beta'] * aux['loss_agent_entropy'].mean()
        loss_opponent = aux['loss_opponent'].mean()
        return jp.stack([loss_agent, loss_opponent]), aux

    grads, aux = jax.jacobian(loss_fn, has_aux=True)(agent)
    grad_agent = jax.tree_map(lambda x: x[0], grads)
    grad_opponent = jax.tree_map(lambda x: x[1], grads)

    def clip_grad(g):
        clip_grad_config = hp['actor']['train']['clip_grad']
        if clip_grad_config['mode'] == 'norm':
            max_norm = clip_grad_config['max_norm']
            g = clip_grads_by_norm(g, max_norm)
        elif clip_grad_config['mode'] == 'disabled':
            pass
        else:
            raise ValueError(f"Unknown grad_clip mode {clip_grad_config['mode']}")
        return g

    if hp['actor']['train']['separate_optimizers'] == 'disabled':
        opt_loss = optimizers['opt_loss']
        if include_opponent:
            grad = jax.tree_map(lambda a, b: a + hp['opponent_differentiation_weight'] * b, grad_agent, grad_opponent)
        else:
            grad = grad_agent

        grad = clip_grad(grad)
        updates, new_opt_loss_state = opt_loss.opt.update(grad, opt_loss.opt_state, agent)
        new_agent = optax.apply_updates(agent, updates)
        new_optimizer_states = {'new_opt_loss_state': new_opt_loss_state}

    elif hp['actor']['train']['separate_optimizers'] == 'enabled':
        opt_loss_agent = optimizers['opt_loss_agent']
        opt_loss_opponent = optimizers['opt_loss_opponent']
        grad_agent = clip_grad(grad_agent)
        grad_opponent = clip_grad(grad_opponent)
        updates_agent, new_opt_loss_agent_state = opt_loss_agent.opt.update(grad_agent, opt_loss_agent.opt_state, agent)
        updates_opponent, new_opt_loss_opponent_state = opt_loss_opponent.opt.update(grad_opponent, opt_loss_opponent.opt_state, agent)
        new_agent = optax.apply_updates(agent, updates_agent)
        new_agent = optax.apply_updates(new_agent, updates_opponent)
        new_optimizer_states = {'new_opt_loss_agent_state': new_opt_loss_agent_state, 'new_opt_loss_opponent_state': new_opt_loss_opponent_state}

    out = {'agent': new_agent,
           **new_optimizer_states,
           'loss_agent_entropy': aux['loss_agent_entropy'],
           'grad_agent_norm': global_norm(grad_agent),
           'grad_opponent_norm': global_norm(grad_opponent)
           }

    return out


def train_agent_actor(state, opponents, hp, episodes, player_to_train: int, include_opponent: bool = False):
    # pop state
    agent = state[f'agent{player_to_train}']
    opponent_player = 1 - player_to_train

    if hp['actor']['train']['separate_optimizers'] == 'enabled':
        opt_loss_agent = state[f'agent{player_to_train}_opt_actor_loss_agent']
        opt_loss_opponent = state[f'agent{player_to_train}_opt_actor_loss_opponent']
        optimizers = {'opt_loss_agent': opt_loss_agent, 'opt_loss_opponent': opt_loss_opponent}
    elif hp['actor']['train']['separate_optimizers'] == 'disabled':
        opt_loss = state[f'agent{player_to_train}_opt_actor_loss']
        optimizers = {'opt_loss': opt_loss}

    aux = update_agent_actor(agent=agent,
                             opponents=opponents,
                             optimizers=optimizers,
                             hp=hp,
                             episodes=episodes,
                             player_to_train=player_to_train,
                             include_opponent=include_opponent, )

    if hp['actor']['train']['separate_optimizers'] == 'enabled':
        state[f'agent{player_to_train}_opt_actor_loss_agent'] = opt_loss_agent.replace(opt_state=aux['new_opt_loss_agent_state'])
        if include_opponent:
            state[f'agent{player_to_train}_opt_actor_loss_opponent'] = opt_loss_opponent.replace(opt_state=aux['new_opt_loss_opponent_state'])
    elif hp['actor']['train']['separate_optimizers'] == 'disabled':
        state[f'agent{player_to_train}_opt_actor_loss'] = opt_loss.replace(opt_state=aux['new_opt_loss_state'])
    state[f'agent{player_to_train}'] = aux['agent']
    return aux


def entropy(logps):
    return -jp.sum(jp.nan_to_num(logps * jp.exp(logps)), axis=-1)


def test_entropy():
    p11, p12, p21, p22 = 0.1, 0.9, 0.2, 0.8
    logps = jp.log(jp.array([[p11, p12], [p21, p22]]))
    ent = entropy(logps)
    assert ent[0] == -jp.sum(jp.array([p11, p12]) * jp.log(jp.array([p11, p12])))
    assert ent[1] == -jp.sum(jp.array([p21, p22]) * jp.log(jp.array([p21, p22])))


@partial(jax.jit, static_argnames=('hp',))
def differentiable_loaded_dice_returns(op_rewards, agent_logps, op_values, hp):
    logps_and_one = jp.concatenate([jp.ones(1), agent_logps])  # [T+1]
    dice = magic_box(jp.cumsum(logps_and_one))  # [T+1]
    dice = dice[1:] - dice[:-1]  # [T]

    op_returns = discounted_returns(op_rewards, hp['reward_discount'])  # [T]

    reward_discount = hp['reward_discount']
    diff_op_discount = hp['differentiable_opponent']['discount']

    def _body(c, x):
        R = x['R']
        d = x['d']
        V = x['V']
        a = (R - V) * d + reward_discount * diff_op_discount * c
        return a, a

    _, op_returns_zero = jax.lax.scan(f=_body, init=0., xs={'R': jp.flip(op_returns), 'd': jp.flip(dice), 'V': jp.flip(op_values[:-1])})
    op_returns_zero = jp.flip(op_returns_zero)
    op_returns = op_returns_zero + op_returns

    return op_returns


@partial(jax.jit, static_argnames='hp')
def agent_policy_loss(agent, opponent, hp, episodes, player_to_train: int, op_rewards_baseline):
    # simple reinforce with baseline
    agent_rewards = episodes['rew'][:, player_to_train]
    returns = get_returns(agent_rewards, hp)  # [T]

    dummy_rng = rax.PRNGKey(0)
    T = episodes['obs'].shape[0] - 1
    player_obs = episodes['obs'][:, player_to_train]
    obs_without_last = player_obs[:-1]
    obs_with_last = player_obs
    agent_actions = episodes['act'][:, player_to_train]

    out = agent.call_seq({'obs_seq': obs_with_last})  # [T, 2]
    all_values_with_last = out['value_seq']
    all_logps = out['logp_seq'][:-1]
    all_values = out['value_seq'][:-1]

    # jax.debug.print("recomputed all logps abs diff with episodes logps: {}", jp.abs(all_logps - episodes['logp'][:, player_to_train]).mean())

    all_values = jax.lax.stop_gradient(all_values)  # [T]
    logps = jax.vmap(lambda lps, act: lps[act])(all_logps, agent_actions)  # [T]
    causal_logps = jp.cumsum(logps)  # [T]
    if hp['actor']['train']['advantage'] == 'MC':
        advantage = returns - all_values
        loss_agent = -(logps * advantage).mean()
    elif hp['actor']['train']['advantage'] == 'TD0':
        advantage = agent_rewards + hp['reward_discount'] * all_values_with_last[1:] - all_values_with_last[:-1]
        loss_agent = -(logps * advantage).mean()
    else:
        raise ValueError(f"Unknown advantage type {hp['actor']['train']['advantage']}")

    loss_agent_entropy = -1. * entropy(all_logps).mean()  # [1

    opponent_player = 1 - player_to_train
    op_rewards = episodes['rew'][:, opponent_player]  # [T]
    op_actions = episodes['act'][:, opponent_player]  # [T]

    opponent_obs = episodes['obs'][:, opponent_player]  # [T+1, 4]
    opponent_obs_without_last = opponent_obs[:-1]
    opponent_obs_with_last = opponent_obs
    op_out = opponent.call_seq({'obs_seq': opponent_obs_with_last})
    op_values = op_out['value_seq']  # [T+1]
    op_values = jax.lax.stop_gradient(op_values)  # [T+1] we are not differentiating w.r.t to the opponent but good practice to stop gradient anyway
    op_qvalues = op_out['qvalue_seq']  # [T+1, 4]

    if hp['differentiable_opponent']['method'] == 'n_step':
        n_step = hp['differentiable_opponent']['n_step']
        op_n_qvalues = n_step_value(op_values, op_rewards, hp['reward_discount'], n_step, logps, reward_baseline=op_rewards_baseline)  # [T+1-n]
        op_qvalues = op_qvalues[:-n_step]  # [T+1-n, 2]

        op_actions = op_actions[:(-n_step + 1)]  # [T+1-n]
        timesteps = jp.arange(op_qvalues.shape[0])  # [T+1-n]

        dif_op_qvalues = op_qvalues.at[timesteps, op_actions].set(op_n_qvalues)  # [T+1-n, 2]
        op_logps = jax.nn.log_softmax(dif_op_qvalues / hp['op_softmax_temp'])  # [T+1-n, 2]
        op_logps = jax.vmap(lambda lps, act: lps[act])(op_logps, op_actions)  # [T+1-n]

        truncated_values = all_values[:(-n_step + 1)]  # [T+1-n]
        if hp['actor']['train']['advantage'] == 'MC':
            truncated_returns = returns[:(-n_step + 1)]  # [T+1-n]
            loss_opponent = -(op_logps * (truncated_returns - truncated_values)).mean()
        elif hp['actor']['train']['advantage'] == 'TD0':
            truncated_rewards = agent_rewards[:(-n_step + 1)]  # [T+1-n]
            truncated_values_with_zero = jp.concatenate([truncated_values, jp.zeros(1)])
            loss_opponent = -(op_logps * (truncated_rewards + hp['reward_discount'] * truncated_values_with_zero[1:] - truncated_values_with_zero[:-1])).mean()

    elif hp['differentiable_opponent']['method'] in ['loaded-dice', 'loaded-ios']:

        if hp['differentiable_opponent']['method'] == 'loaded-dice':
            op_returns = differentiable_loaded_dice_returns(op_rewards=op_rewards, agent_logps=logps, op_values=op_values, hp=hp)

            if hp['differentiable_opponent']['differentiable_current_reward']:
                op_differentiable_qvalues = op_returns
            else:
                op_differentiable_qvalues = op_rewards[:-1] + hp['reward_discount'] * op_returns[1:]
                op_differentiable_qvalues = jp.concatenate([op_differentiable_qvalues, op_rewards[-1:]])

            timesteps = jp.arange(op_differentiable_qvalues.shape[0])  # [T]
            dif_op_qvalues = op_qvalues[:-1].at[timesteps, op_actions].set(op_differentiable_qvalues)  # [T, 4]
            op_logps = jax.nn.log_softmax(dif_op_qvalues / hp['op_softmax_temp'])  # [T, 4]
            op_logps = jax.vmap(lambda lps, act: lps[act])(op_logps, op_actions)  # [T]
            loss_opponent = -(op_logps * advantage).mean()

        elif hp['differentiable_opponent']['method'] == 'loaded-ios':
            inf_weight = hp['actor']['train']['inf_weight']
            op_returns = differentiable_loaded_dice_returns(op_rewards=op_rewards, agent_logps=logps, op_values=op_values, hp=hp)
            op_advantage = op_rewards + hp['reward_discount'] * op_values[1:] - op_values[:-1]
            n_advantage = (advantage - jp.mean(advantage)) / jp.std(advantage)
            n_op_advantage = (op_advantage - jp.mean(op_advantage)) / jp.std(op_advantage)
            mask = -1 * jp.where(n_op_advantage > 0, 1., 0.) * jp.where(n_advantage < 0, 1., 0.)
            loss_opponent = inf_weight * (op_returns * advantage * mask).mean()

    else:
        raise ValueError(f"Unknown differentiable opponent method {hp['differentiable_opponent']['method']}")

    return {'loss_agent': loss_agent,
            'loss_opponent': loss_opponent,
            'loss_agent_entropy': loss_agent_entropy}


def n_step_value(values, rewards, gamma, n, logps=None, reward_baseline=0):
    assert values.shape[0] == rewards.shape[0] + 1
    assert len(rewards.shape) == 1
    assert len(values.shape) == 1
    if logps is not None:
        assert logps.shape == rewards.shape

    def fun(t):
        rs = jax.lax.dynamic_slice_in_dim(rewards, t, n, axis=0)
        end_state_value = values[t + n]

        # if logps are provided, we use the magic box trick to make the n-step return of the opponent differentiable w.r.t to the agent's policy
        if logps is not None:
            lps = jax.lax.dynamic_slice_in_dim(logps, t, n, axis=0)
            lps = lps.at[0].set(jax.lax.stop_gradient(lps[0]))  # we don't want the action at state at time 't' to be differentiable as it is taken not aware of the opponent's action
            causal_lps = lps.cumsum()

            rs = (rs - reward_baseline) * magic_box(causal_lps) + reward_baseline

        return discounted_returns(rs, discount_factor=gamma)[0] + end_state_value * gamma ** n

    T = values.shape[0] - 1
    truncated_times = jp.arange(T + 1 - n)
    return jax.vmap(fun)(truncated_times)


def test_n_step_value():
    v0, v1, v2, v3, v4, v5, v6 = 6, 5, 9, 10, 5, 7, 8
    r0, r1, r2, r3, r4, r5 = +1, -2, +3, -1, +1, +2
    values = jp.array([v0, v1, v2, v3, v4, v5, v6])
    rewards = jp.array([r0, r1, r2, r3, r4, r5])
    gamma = 0.9
    n = 3
    out = n_step_value(values, rewards, gamma, n)
    gold_out = jp.array([r0 + gamma * r1 + gamma ** 2 * r2 + gamma ** 3 * v3,
                         r1 + gamma * r2 + gamma ** 2 * r3 + gamma ** 3 * v4,
                         r2 + gamma * r3 + gamma ** 2 * r4 + gamma ** 3 * v5,
                         r3 + gamma * r4 + gamma ** 2 * r5 + gamma ** 3 * v6])
    assert jp.allclose(out, gold_out), f'{out} != {gold_out}'


@partial(jax.jit, static_argnames=('hp',))
def update_agent_qvalue(agent, agent_ema, agent_opt, hp, episodes, player_to_train: int):
    def loss_fn(a):
        aux = jax.vmap(lambda ep: agent_qvalue_loss(a, agent_ema, hp, ep, player_to_train))(episodes)
        return aux['qvalue_loss'].mean(), aux

    grad, aux = jax.grad(loss_fn, has_aux=True)(agent)
    updates, new_agent_opt_state = agent_opt.opt.update(grad, agent_opt.opt_state, agent)
    agent = optax.apply_updates(agent, updates)
    return {'agent': agent, 'new_opt_state': new_agent_opt_state, **aux}


def train_agent_qvalue(state, hp, episodes, player_to_train: int):
    agent_ema = state[f'agent{player_to_train}_ema']
    agent = state[f'agent{player_to_train}']
    agent_opt = state[f'agent{player_to_train}_opt_qvalue']
    aux = update_agent_qvalue(agent, agent_ema, agent_opt, hp, episodes, player_to_train)
    new_agent = aux['agent']
    state[f'agent{player_to_train}'] = new_agent
    state[f'agent{player_to_train}_opt_qvalue'] = agent_opt.replace(opt_state=aux['new_opt_state'])
    ema_gamma = hp['qvalue']['train']['target_ema_gamma']
    agent_ema_params = jax.tree_map(lambda old, new: ema_gamma * old + (1 - ema_gamma) * new, agent_ema.params, new_agent.params)
    state[f'agent{player_to_train}_ema'] = agent_ema.replace(params=agent_ema_params)

    return aux


def huber_loss(x, delta: float = 1.):
    # taken from https://github.com/deepmind/rlax/blob/f1ad41f79d617551911da4fd61acca99d8fea84c/rlax/_src/clipping.py
    # 0.5 * x^2                  if |x| <= d
    # 0.5 * d^2 + d * (|x| - d)  if |x| > d
    abs_x = jp.abs(x)
    quadratic = jp.minimum(abs_x, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_x - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def agent_qvalue_loss(agent, agent_ema, hp, episodes, player_to_train: int):
    T = episodes['obs'].shape[0] - 1
    rewards = episodes['rew'][:, player_to_train]
    player_obs = episodes['obs'][:, player_to_train].reshape(T + 1, -1)
    obs_without_last = player_obs[:-1]
    obs_with_last = player_obs

    out_seq = agent.call_seq({'obs_seq': obs_without_last})
    all_qvalues = out_seq['qvalue_seq']
    qvalues = jax.vmap(lambda qs, act: qs[act[player_to_train]])(all_qvalues, episodes['act'])

    ema_out_seq = agent_ema.call_seq({'obs_seq': obs_with_last})
    ema_all_qvalues = ema_out_seq['qvalue_seq']  # [T + 1, num_actions]

    if hp['qvalue']['mode'] == 'argmax':
        ema_qvalues = ema_all_qvalues.max(axis=-1)
    elif hp['qvalue']['mode'] == 'mean':
        ema_all_qvalues = ema_all_qvalues[:-1, :]  # [T, num_actions]
        ema_qvalues = jax.vmap(lambda qs, act: qs[act[player_to_train]])(ema_all_qvalues, episodes['act'])  # [T]
        ema_last_state_value = ema_out_seq['value_seq'][-1].reshape(1)  # [1]
        ema_qvalues = jp.concatenate([ema_qvalues, ema_last_state_value], axis=0)  # [T+1]
    else:
        raise ValueError(f'Unknown qvalue mode {hp["qvalue"]["mode"]}')

    target = rewards + hp['reward_discount'] * ema_qvalues[1:]
    target = jax.lax.stop_gradient(target)

    qvalue_loss = huber_loss(qvalues - target).mean()

    return {'qvalue_loss': qvalue_loss}


def discounted_returns(rewards, discount_factor):
    def _body(c, r):
        a = r + discount_factor * c
        return a, a

    _, returns = jax.lax.scan(f=_body, init=0., xs=jp.flip(rewards))
    return jp.flip(returns)


def get_returns(rewards, hp):
    returns = discounted_returns(rewards, hp['reward_discount'])
    return returns


def test_get_returns():
    rewards = jp.array([1, 2, 3])
    gamma = 0.6
    hp = {'reward_discount': gamma}
    returns = get_returns(rewards, hp)
    gold_returns = jp.array([1 + gamma * 2 + (gamma ** 2) * 3, 2 + gamma * 3, 3.])
    assert jp.allclose(returns, gold_returns), f'{returns} != {gold_returns}'


def qvalue_policy_divergence(qvalues, logps):
    p_qvalues = jax.nn.softmax(qvalues)
    p = jp.exp(logps)
    kl_q_logps = (p * (logps - jp.log(p_qvalues))).sum()
    kl_logps_q = (p_qvalues * (jp.log(p_qvalues) - logps)).sum()
    return {'kl_q_logps': kl_q_logps, 'kl_logps_q': kl_logps_q}


def episodes_qvalue_policy_divergence(state, episodes, hp):
    def episode_qvalue_policy_divergence(agent, ep, player_id: int):
        dummy_rng = rax.PRNGKey(0)
        obs_without_last = ep['obs'][:, player_id]

        if hp['actor']['model'] in ['mlp', 'logits']:
            all_qvalues = jax.vmap(lambda obs, t: agent({'obs': obs, 'rng': dummy_rng, 't': t})['qvalue'])(obs_without_last, jp.arange(obs_without_last.shape[0]))
            all_logps = jax.vmap(lambda obs, t: agent({'obs': obs, 'rng': dummy_rng, 't': t})['logp'])(obs_without_last, jp.arange(obs_without_last.shape[0]))
        elif hp['actor']['model'] in ['gru']:
            out_seq = agent.call_seq({'obs_seq': obs_without_last})
            all_qvalues = out_seq['qvalue_seq']
            all_logps = out_seq['logp_seq']

        aux = qvalue_policy_divergence(all_qvalues, all_logps)
        return aux

    aux0 = jax.vmap(lambda ep: episode_qvalue_policy_divergence(state['agent0'], ep, player_id=0))(episodes)
    aux1 = jax.vmap(lambda ep: episode_qvalue_policy_divergence(state['agent1'], ep, player_id=1))(episodes)
    return {'kl_q_logps_0': aux0['kl_q_logps'].mean(), 'kl_logps_q_1': aux0['kl_logps_q'].mean(),
            'kl_q_logps_1': aux1['kl_q_logps'].mean(), 'kl_logps_q_0': aux1['kl_logps_q'].mean()}


@jax.jit
def tree_stack(xs):
    return jax.tree_map(lambda *args: jp.stack(args), *xs)

@partial(jax.jit, static_argnames=('B',))
def tree_unstack(xs, B):
    episodes = [jax.tree_map(lambda x: x[i], xs) for i in range(B)]
    return episodes

@partial(jax.jit, static_argnames=('hp',))
def sample_agent_params(hp, rb, agent, rb_rng, min_valid_index: int):
    rb_size = hp['agent_replay_buffer']['capacity']
    # indicate sizes of replay buffer and agent
    B = hp['batch_size']
    cur_agent_frac = hp['agent_replay_buffer']['cur_agent_frac']
    cur_agent_size = int(B * cur_agent_frac)
    sample_size = B - cur_agent_size

    agent_indices = jax.random.randint(rb_rng, shape=(sample_size,), minval=min_valid_index, maxval=rb_size)
    sample_params = jax.tree_map(lambda x: x[agent_indices], rb)
    agent_params = jax.vmap(lambda i: agent.params)(jp.arange(cur_agent_size))
    final_agent_params = jax.tree_map(lambda *xs: jp.concatenate(xs, axis=0), sample_params, agent_params)

    return final_agent_params

class EpisodeReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, episodes):
        B = episodes['obs'].shape[0]
        episodes = tree_unstack(episodes, B)
        self.buffer.extend(episodes)

    def sample(self, batch_size):
        episodes = random.choices(self.buffer, k=batch_size)  # sample with replacement as the buffer may be smaller than batch_size at the beginning
        episodes = tree_stack(episodes)
        return episodes

@jax.jit
def push_to_rb(rb, params):
    to_keep_rb = jax.tree_map(lambda x: x[1:], rb)
    params = jax.tree_map(lambda x: x[None], params)
    new_rb = jax.tree_map(lambda x, y: jp.concatenate((x, y), axis=0), to_keep_rb, params)
    return new_rb

def use_rb(hp):
    return hp['agent_replay_buffer']['mode'] == 'enabled'

def setup_state(hp):
    just_self_play = hp['just_self_play']

    if just_self_play:
        state = AliasDict(redirects={
            'agent1': 'agent0',
            'rb_agent1_params': 'rb_agent0_params',
        })
    else:
        state = dict()

    state['rng'] = rax.PRNGKey(hp['seed'])

    state['rng'], rng0, rng1 = rax.split(state['rng'], 3)
    dummy_obs = jp.zeros(4)
    dummy_obs_seq = jp.zeros((hp['game_length'], 4))

    if hp['actor']['model'] in ['mlp', 'logits']:
        agent_module = OneStepIPDAgentDense(hp['actor'])
        agent0_params = agent_module.init(rng0, {'obs': dummy_obs, 'rng': rng_dummy, 't': 0})
        if not just_self_play:
            agent1_params = agent_module.init(rng1, {'obs': dummy_obs, 'rng': rng_dummy, 't': 0})
    elif hp['actor']['model'] in ['gru']:
        agent_module = GRUIPDAgent()
        agent0_params = agent_module.init(rng0, {'obs_seq': dummy_obs_seq, 'rng': rng_dummy, 't': 0})
        if not just_self_play:
            agent1_params = agent_module.init(rng1, {'obs_seq': dummy_obs_seq, 'rng': rng_dummy, 't': 0})
    else:
        raise ValueError(f'Unknown actor model: {hp["actor"]["model"]}')

    agent0 = IPDAgent(agent0_params, agent_module)
    state['agent0'] = agent0
    if not just_self_play:
        agent1 = IPDAgent(agent1_params, agent_module)
        state['agent1'] = agent1

    # --- defining replay buffers ---
    def create_rb_agent_params(player_id: int):
        rb_size = hp['agent_replay_buffer']['capacity']
        tmp_rb = [state[f'agent{player_id}'].params for _ in range(rb_size)]
        state[f'rb_agent{player_id}_params'] = jax.tree_map(lambda *xs: jp.stack(xs, axis=0), *tmp_rb)
        state['min_valid_index_rb'] = rb_size  # first, the buffer is not valid

    if use_rb(hp):
        create_rb_agent_params(player_id=0)
        if not just_self_play:
            create_rb_agent_params(player_id=1)

    # --- defining ema ---
    state['agent0_ema'] = agent0
    if not just_self_play:
        state['agent1_ema'] = agent1

    if hp['actor']['train']['separate_optimizers'] == 'enabled':
        if hp['optimizer'] == 'adam':
            agent_opt = optax.adam(hp['actor']['train']['lr_loss_agent'])
            opponent_opt = optax.adam(hp['actor']['train']['lr_loss_opponent'])
        elif hp['optimizer'] == 'sgd':
            agent_opt = optax.sgd(hp['actor']['train']['lr_loss_agent'])
            opponent_opt = optax.sgd(hp['actor']['train']['lr_loss_opponent'])
        else:
            raise ValueError(f'Unknown optimizer {hp["optimizer"]}')
        state['agent0_opt_actor_loss_agent'] = Optimizer(agent_opt, agent_opt.init(agent0))
        state['agent0_opt_actor_loss_opponent'] = Optimizer(opponent_opt, opponent_opt.init(agent0))
        if not just_self_play:
            state['agent1_opt_actor_loss_agent'] = Optimizer(agent_opt, agent_opt.init(agent1))
            state['agent1_opt_actor_loss_opponent'] = Optimizer(opponent_opt, opponent_opt.init(agent1))
    elif hp['actor']['train']['separate_optimizers'] == 'disabled':
        if hp['optimizer'] == 'adam':
            opt = optax.adam(hp['actor']['train']['lr_loss_actor'])
        elif hp['optimizer'] == 'sgd':
            opt = optax.sgd(hp['actor']['train']['lr_loss_actor'])
        else:
            raise ValueError(f'Unknown optimizer {hp["optimizer"]}')
        state['agent0_opt_actor_loss'] = Optimizer(opt, opt.init(agent0))
        if not just_self_play:
            state['agent1_opt_actor_loss'] = Optimizer(opt, opt.init(agent1))
    else:
        raise ValueError(f'Unknown separate_optimizers mode {hp["actor"]["train"]["separate_optimizers"]}')

    qvalue_opt = optax.adam(hp['qvalue']['train']['lr_loss_qvalue'])
    state['agent0_opt_qvalue'] = Optimizer(qvalue_opt, qvalue_opt.init(agent0))
    if not just_self_play:
        state['agent1_opt_qvalue'] = Optimizer(qvalue_opt, qvalue_opt.init(agent1))
    return state


def train(hp, log_wandb):
    start_time = time.time()
    hp = flax.core.FrozenDict(hp)
    just_self_play = hp['just_self_play']
    if just_self_play:
        print('*** JUST SELF PLAY ***')
    print('actor:', hp['actor'])

    if hp['qvalue']['replay_buffer']['mode'] == 'disabled':
        episodes_replay_buffer = None
    elif hp['qvalue']['replay_buffer']['mode'] == 'enabled':
        episodes_replay_buffer_0 = EpisodeReplayBuffer(capacity=hp['qvalue']['replay_buffer']['capacity'])
        if not just_self_play:
            episodes_replay_buffer_1 = EpisodeReplayBuffer(capacity=hp['qvalue']['replay_buffer']['capacity'])
    else:
        raise ValueError(f'Unknown replay buffer mode: {hp["qvalue"]["replay_buffer"]["mode"]}')

    state = setup_state(hp)

    agent0_params = state['agent0'].params  # params for reset
    if not just_self_play:
        agent1_params = state['agent1'].params  # params for reset



    for i in range(500000):

        # --- updating replay buffers ---
        if use_rb(hp) and i % hp['agent_replay_buffer']['update_freq'] == 0:
            def update_agent_rb(player_id: int):
                rb = state[f'rb_agent{player_id}_params']
                agent_params = state[f'agent{player_id}'].params
                rb = push_to_rb(rb, agent_params)
                state[f'rb_agent{player_id}_params'] = rb

            update_agent_rb(player_id=0)
            update_agent_rb(player_id=1)
            state['min_valid_index_rb'] = max(0, state['min_valid_index_rb'] - 1)

        # --- sampling from replay buffers ---

        if use_rb(hp):
            def sample_agents(player_id: int):
                agent_params = sample_agent_params(hp=hp,
                                                   rb=state[f'rb_agent{player_id}_params'],
                                                   agent=state[f'agent{player_id}'],
                                                   rb_rng=rscope(state['rng'], f'rb{player_id}'),
                                                   min_valid_index=state['min_valid_index_rb'])
                agents = jax.vmap(lambda p: state[f'agent{player_id}'].replace(params=p))(agent_params)
                return agents

            if just_self_play:
                samples = {
                    'agent1s': sample_agents(player_id=1),  # these will actually be agent0s because of the aliasing
                }
            else:
                samples = {
                    'agent0s': sample_agents(player_id=0),
                    'agent1s': sample_agents(player_id=1),
                }

        # --- generating episodes ---
        def gen_episode(player_id: int):
            return gen_episodes(agent=state[f'agent{player_id}'],
                                opponents=samples[f'agent{1 - player_id}s'] if use_rb(hp) else [state[f'agent{1 - player_id}']],
                                rng=rscope(state['rng'], f'episode{player_id}'),
                                hp=hp)

        episodes = {0: gen_episode(player_id=0)}
        if not just_self_play:
            episodes[1] = gen_episode(player_id=1)

        # ---- actor training ----
        update = train_agent_actor(state=state,
                                   opponents = samples['agent1s'] if use_rb(hp) else [state['agent1']],
                                   hp=hp,
                                   episodes=episodes[0],
                                   player_to_train=0,
                                   include_opponent=True)
        agent0_entropy = -1 * update['loss_agent_entropy']
        grad_agent0_norm = update['grad_agent_norm']
        grad_opponent0_norm = update['grad_opponent_norm']
        if not just_self_play:
            update = train_agent_actor(state=state,
                                       opponents = samples['agent0s'] if use_rb(hp) else [state['agent0']],
                                       hp=hp,
                                       episodes=episodes[1],
                                       player_to_train=1,
                                       include_opponent=True)
            agent1_entropy = -1 * update['loss_agent_entropy']
            grad_agent1_norm = update['grad_agent_norm']
            grad_opponent1_norm = update['grad_opponent_norm']
        # ---- qvalue training ----

        if hp['qvalue']['replay_buffer']['mode'] == 'enabled':
            episodes_replay_buffer_0.push(episodes[0])
            if not just_self_play:
                episodes_replay_buffer_1.push(episodes[1])
            episodes_for_qvalue_0 = episodes_replay_buffer_0.sample(hp['batch_size'])
            if not just_self_play:
                episodes_for_qvalue_1 = episodes_replay_buffer_1.sample(hp['batch_size'])
        else:
            episodes_for_qvalue_0 = episodes[0]
            if not just_self_play:
                episodes_for_qvalue_1 = episodes[1]

        update = train_agent_qvalue(state, hp, episodes_for_qvalue_0, player_to_train=0)
        loss_agent_0_qvalue = update['qvalue_loss']
        if not just_self_play:
            update = train_agent_qvalue(state, hp, episodes_for_qvalue_1, player_to_train=1)
            loss_agent_1_qvalue = update['qvalue_loss']

        # ---- reset agents ----
        if hp['reset']['mode'] == 'disabled':
            pass
        elif hp['reset']['mode'] == 'enabled':
            if i % hp['reset']['every'] == 0:
                state['agent0'] = state['agent0'].replace(params=agent0_params)
                if not just_self_play:
                    state['agent1'] = state['agent1'].replace(params=agent1_params)

        # --- log everything ---

        if i % hp['eval_every'] == 0:
            print(f'iteration {i}')
            update = episodes_qvalue_policy_divergence(state, episodes[0], hp)
            print(f'kl_q_logps_0: {update["kl_q_logps_0"]:.3f}, kl_logps_1_0: {update["kl_logps_q_1"]:.3f}, '
                  f'kl_q_logps_1: {update["kl_q_logps_1"]:.3f}, kl_logps_q_0: {update["kl_logps_q_0"]:.3f}')
            print(f'loss_agent_0_qvalue: {loss_agent_0_qvalue.mean():.3f}')
            print(f'agent0_entropy: {agent0_entropy.mean():.3f}')
            print(f'grad_agent0_norm: {grad_agent0_norm.mean():.3f}, grad_opponent0_norm: {grad_opponent0_norm.mean():.3f}')
            if not just_self_play:
                print(f'loss_agent_1_qvalue: {loss_agent_1_qvalue.mean():.3f}')
                print(f'agent1_entropy: {agent1_entropy.mean():.3f}')
                print(f'grad_agent1_norm: {grad_agent1_norm.mean():.3f}, grad_opponent1_norm: {grad_opponent1_norm.mean():.3f}')

            if log_wandb:
                logs = {
                    'step': i,
                    'walltime': time.time() - start_time,
                    'kl_q_logps_0': update['kl_q_logps_0'],
                    'kl_logps_q_1': update['kl_logps_q_1'],
                    'kl_q_logps_1': update['kl_q_logps_1'],
                    'kl_logps_q_0': update['kl_logps_q_0'],
                    'loss_agent_0_qvalue': loss_agent_0_qvalue.mean(),
                    'agent0_entropy': agent0_entropy.mean(),
                    'grad_agent0_norm': grad_agent0_norm.mean(),
                    'grad_opponent0_norm': grad_opponent0_norm.mean(),
                    'agent0_mean_rewards': episodes[0]['rew'][..., 0].mean(),
                }
                wandb.log(logs, step=i)
                if not just_self_play:
                    logs_1 = {
                        'loss_agent_1_qvalue': loss_agent_1_qvalue.mean(),
                        'agent1_entropy': agent1_entropy.mean(),
                        'grad_agent1_norm': grad_agent1_norm.mean(),
                        'grad_opponent1_norm': grad_opponent1_norm.mean(),
                        'agent1_mean_rewards': episodes[1]['rew'][..., 1].mean(),
                    }
                    wandb.log({**logs, **logs_1}, step=i)

            states = {'START': jp.zeros(4),
                      'CC': jp.array([1, 0, 1, 0]),
                      'CD': jp.array([1, 0, 0, 1]),
                      'DC': jp.array([0, 1, 1, 0]),
                      'DD': jp.array([0, 1, 0, 1])}

            if hp['actor']['model'] in ['mlp', 'logits']:
                for player in [0, 1]:
                    for obs_name, obs in states.items():
                        dummy_rng = rax.PRNGKey(0)
                        qvalue = state[f'agent{player}']({'obs': obs, 'rng': dummy_rng, 't': 0})['qvalue']
                        value = state[f'agent{player}']({'obs': obs, 'rng': dummy_rng, 't': 0})['value']
                        logp = state[f'agent{player}']({'obs': obs, 'rng': dummy_rng, 't': 0})['logp']
                        p = jp.exp(logp)
                        print(f'state: {obs_name}, qvalue_{player}: {qvalue}, p_{player}: {p} value_{player}: {value}')
                        if log_wandb:
                            wandb.log({f'qvalue_{player}_{obs_name}_C': qvalue[0],
                                       f'qvalue_{player}_{obs_name}_D': qvalue[1],
                                       f'p_{player}_{obs_name}_C': p[0],
                                       f'p_{player}_{obs_name}_D': p[1],
                                       f'value_{player}_{obs_name}': value})

            obs_sample = episodes[0]['obs'][0, :, 0]  # first episode, first agent

            def obs_to_state_names(obs):
                state_names = []
                for t in range(obs.shape[0]):
                    s = obs[t]
                    # using states dict from above to get the state name
                    state_name = [k for k, v in states.items() if (v == s).all()][0]
                    state_names.append(state_name)
                return state_names

            state_names_sample = obs_to_state_names(obs_sample)
            counter = Counter(state_names_sample)
            print(f"sample game: {counter}")
            print(f"sample game: {' '.join(state_names_sample)}")
            B = episodes[0]['obs'].shape[0]
            batch_state_name = [obs_to_state_names(episodes[0]['obs'][b, :, 0]) for b in range(min(B, 128))]

            state_transition_array = np.zeros((2, 5, 2))
            # analyse policies
            for player in [0, 1]:

                for state_name_seq in batch_state_name:
                    for idx in range(len(state_name_seq) - 1):
                        cur = state_name_seq[idx]
                        cur = {'START': 0, 'CC': 1, 'CD': 2, 'DC': 3, 'DD': 4}[cur]
                        next = state_name_seq[idx + 1][player]  # to get the action of the player at that state
                        next = {'C': 0, 'D': 1}[next]
                        state_transition_array[player, cur, next] += 1

                state_transition_array[player] /= state_transition_array[player].sum(axis=1, keepdims=True)

            # print the state transition matrix with the corresponding state names
            wandb_cache = {}
            for player in [0, 1]:
                print(f'player {player} state transition matrix')
                for idx, state_name in enumerate(['START', 'CC', 'CD', 'DC', 'DD']):
                    print(f'{state_name}: {state_transition_array[player, idx]}')
                    # update wandb cache
                    wandb_cache[f'{player}_{state_name}_C'] = state_transition_array[player, idx, 0]
            if log_wandb:
                wandb.log(wandb_cache, step=i)
            print(f'Episodes 0: agent0 mean rewards: {episodes[0]["rew"][..., 0].mean():.3f}, agent1 mean rewards: {episodes[0]["rew"][..., 1].mean():.3f}')
            if not just_self_play:
                print(f'Episodes 1: agent0 mean rewards: {episodes[1]["rew"][..., 0].mean():.3f}, agent1 mean rewards: {episodes[1]["rew"][..., 1].mean():.3f}')


def run_tests():
    test_entropy()
    test_get_returns()
    test_n_step_value()


@hydra.main(version_base=None, config_path="conf/ipd_conf", config_name="ipd_config")
def main(cfg: DictConfig) -> None:
    jp.set_printoptions(precision=3)
    if cfg.hp.differentiable_opponent.method == 'n_step':
        assert cfg.hp.differentiable_opponent.n_step != 1, 'n_step=1 breaks the logic of the code, especially the slicing [:(-1+1)] returns [] where we want the whole array'

    hp = OmegaConf.to_container(cfg.hp, resolve=True)  # Converts cfg to a Python dict
    print(OmegaConf.to_yaml(cfg.hp))  # Prints the hyperparameters

    log_wandb = cfg.wandb.state == 'enabled'
    if log_wandb:
        wandb_id = wandb.util.generate_id()
        wandb.init(project="loqa-ipd", id=wandb_id, tags=cfg.wandb.tags)
        wandb.config.update(hp)
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

        # go recursively to ./conf and its subdirectories and save every file with yaml
        for root, dirs, files in os.walk('conf'):
            for file in files:
                if file.endswith('.yaml'):
                    wandb.save(os.path.join(root, file))
        # zip the conf folder and save that too
        shutil.make_archive('conf', 'zip', 'conf')
        wandb.save('conf.zip')
        wandb.run.summary.update(slurm_infos())

    train(hp, log_wandb)


if __name__ == '__main__':
    # run_tests()
    main()
