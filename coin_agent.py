from typing import Any

from flax import linen as nn, struct
from jax import numpy as jp, random as rax

from ipd import POLAGRU


class GRUCoinAgent(nn.Module):
    hidden_size_actor: int
    hidden_size_qvalue: int
    layers_before_gru_actor: int
    layers_before_gru_qvalue: int

    def setup(self):
        self.actor_head = POLAGRU(4, self.hidden_size_actor, self.layers_before_gru_actor)
        self.qvalue_head = POLAGRU(4, self.hidden_size_qvalue, self.layers_before_gru_qvalue)

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
        logp_seq = self.logp_seq(x)['logp_seq']  # (T, 4)
        qvalue_seq = self.qvalue_seq(x)['qvalue_seq']  # (T, 4)
        value_seq = (jp.exp(logp_seq) * qvalue_seq).sum(axis=-1)  # (T,)
        return {'logp_seq': logp_seq, 'qvalue_seq': qvalue_seq, 'value_seq': value_seq}

    def logp_seq(self, x):
        obs_seq = x['obs_seq']  # (T, 4)
        logits_seq = self.actor_head(obs_seq, carry=None)['hs']  # (T, 4)
        logp_seq = nn.log_softmax(logits_seq, axis=-1)  # (T, 4)
        return {'logp_seq': logp_seq}

    def qvalue_seq(self, x):
        obs_seq = x['obs_seq']  # (T, 4)
        t_seq = jp.arange(obs_seq.shape[0])  # (T,)
        t_seq = t_seq.reshape(-1, 1)  # (T, 1)
        h_seq = jp.concatenate([obs_seq, t_seq], axis=-1)  # (T, 5)
        qvalue_seq = self.qvalue_head(h_seq, carry=None)['hs']  # (T, 4)
        return {'qvalue_seq': qvalue_seq}

    def call_step(self, x):
        obs = x['obs']  # (T, 4)
        rng = x['rng']
        t = x['t']
        carry_actor = x['carry_actor']
        carry_qvalue = x['carry_qvalue']
        out_actor = self.logp_step({'obs': obs, 'carry': carry_actor})  # (4,)
        logp = out_actor['logp']  # (2,)
        next_carry_actor = out_actor['carry']  # (7,)
        action = rax.categorical(rng, logp)  # (1,)
        out_qvalue = self.qvalue_step({'t': t, 'obs': obs, 'carry': carry_qvalue})  # (4,)
        qvalue = out_qvalue['qvalue']  # (4,)
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
        logits = actor_res['hs'][0]  # (4,)
        logp = nn.log_softmax(logits, axis=-1)  # (4,)
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


@struct.dataclass
class CoinAgent:
    params: Any
    model: Any = struct.field(pytree_node=False)
    player: int = struct.field(pytree_node=False)

    def __call__(self, *args, **kwargs):
        return self.model.apply(self.params, *args, **kwargs)

    def __getattr__(self, name):
        model = self.__dict__.get('model')
        if hasattr(model, name):
            method = getattr(model, name)
            def method_wrapper(*args, **kwargs):
                return model.apply(self.params, *args, **kwargs, method=method)

            return method_wrapper

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get_action(self, rng, episode, t, **kwargs):
        obs_without_last = episode['obs'][:-1, self.player]
        T = obs_without_last.shape[0]
        obs_without_last = obs_without_last.reshape(T, -1)
        logp_seq = self.logp_seq({'obs_seq': obs_without_last})['logp_seq']  # (T, 4)
        action = rax.categorical(rng, logp_seq[t])  # (1,)
        return action
