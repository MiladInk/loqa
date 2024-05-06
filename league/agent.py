from typing import List

import jax
import numpy as np
from flax import linen as nn, struct as struct
from jax import random as rax, numpy as jp

import league.treeanalyser as ta
from coin import make_zero_episode, simplify_observations, coin_taking_summary_matrix, CoinGame, MOVES
from league._utils import FILM, MLP, MLPResidual, MLPResidualLayerNorm, POLAGRU, rscope


class ConvAgent(nn.Module):
    coin_game: "Any" = struct.field(pytree_node=False)
    player: int
    conv_aggregator_activation: "Any"
    aggregator_type: "Any"
    horizon: int
    preprocess_obs_config: "Any"
    value_mlp_features: List[int]
    actor_mlp_features: List[int]
    aggregator_mlp_features: List[int]
    use_film_in_value_for_time: bool
    film_size: int
    hidden_size: int = 128
    normalize: int = 1

    def setup(self):
        if self.conv_aggregator_activation == 'relu':
            activation = nn.relu
        elif self.conv_aggregator_activation == 'tanh':
            activation = nn.tanh
        else:
            raise ValueError(f"Unknown activation {self.conv_aggregator_activation}")
        if self.aggregator_type == 'noagg':
            self.aggregator = lambda h: h  # ConvAggregator(size=self.hidden_size, window=self.horizon, normalize=self.normalize, activation=activation)
        elif self.aggregator_type == 'convagg':
            self.aggregator = ConvAggregator(size=self.hidden_size, window=self.horizon, normalize=self.normalize, activation=activation)
        elif self.aggregator_type == 'mlp':
            self.aggregator = MLP(self.aggregator_mlp_features)
        elif self.aggregator_type == 'mlp-relu':
            self.aggregator = MLP(self.aggregator_mlp_features, last_activation=True)
        elif self.aggregator_type == 'mlp-residual-fixed':
            self.aggregator = MLPResidual(features=self.aggregator_mlp_features)
        elif self.aggregator_type == 'mlp-residual-layernorm':
            self.aggregator = MLPResidualLayerNorm(features=self.aggregator_mlp_features)
        elif self.aggregator_type == 'pola_gru_1':
            self.aggregator = POLAGRU(num_outputs=self.hidden_size, context_size=self.hidden_size, layers_before_gru=1)
        elif self.aggregator_type == 'pola_gru_2':
            self.aggregator = POLAGRU(num_outputs=self.hidden_size, context_size=self.hidden_size, layers_before_gru=2)
        else:
            raise "aggregator not supported"

        self.actor = MLP(self.actor_mlp_features)
        self.value_function = MLP(self.value_mlp_features)
        self.value_function_self_play = MLP(self.value_mlp_features)
        if self.use_film_in_value_for_time:
            self.film_t = FILM(size=self.film_size)
            self.film_t_self_play = FILM(size=self.film_size)

    def preprocess_player_obs(self, episode):
        return self.coin_game.preprocess_player_obs(episode=episode,
                                                    player=self.player,
                                                    horizon=self.horizon,
                                                    config=self.preprocess_obs_config,
                                                    )

    def observe(self, episode):
        h = self.preprocess_player_obs(episode)  # [time, features]
        return self.aggregate(h)

    def aggregate(self, h):
        h = self.aggregator(h)
        if self.aggregator_type.startswith('pola_gru'):
            h = h['hs']  # just return the hidden state sequence
        return h

    def observe_with_carry(self, episode, carry):
        """
        raw version of observe, receives carry, returns full hidden states sequence and carries
        """
        assert self.aggregator_type.startswith('pola_gru'), "Only POLA GRU supports carry"
        obs_seq = self.preprocess_player_obs(episode)
        return self.aggregate_with_carry(obs_seq=obs_seq, carry=carry)

    def aggregate_with_carry(self, obs_seq, carry):
        assert self.aggregator_type.startswith('pola_gru'), "Only POLA GRU supports carry"
        return self.aggregator(x=obs_seq, carry=carry)

    def single_observe_with_carry(self, episode, carry_0_t, t):
        assert self.aggregator_type.startswith('pola_gru'), "Only POLA GRU supports carry"
        if self.preprocess_obs_config['mode'] == 'raw_flat':
            obs_t = episode["obs"][t, self.player].reshape(-1)
        else:
            obs_t = self.preprocess_player_obs(episode)[t]  # [features]
        return self.single_aggregate_with_carry(obs_t, carry_0_t)

    def single_aggregate_with_carry(self, obs_t, carry_0_t):
        assert self.aggregator_type.startswith('pola_gru'), "Only POLA GRU supports carry"
        assert len(obs_t.shape) == 1
        obs = obs_t[None]
        out = self.aggregator(x=obs, carry=carry_0_t)

        hs = out['hs']
        assert hs.shape[0] == 1 and len(hs.shape) == 2
        h = hs[0]
        carry_0_tp1 = out['carry']

        return {'h': h, 'carry': carry_0_tp1}

    def get_single_action_with_carry(self, rng, episode, carry_0_t, t):
        assert self.aggregator_type.startswith('pola_gru'), "Only POLA GRU supports carry"
        out = self.single_observe_with_carry(episode, carry_0_t, t)
        hiddens_t = out['h']
        carry_0_tp1 = out['carry']
        action = self.emit(hiddens_t, rng)
        return action, carry_0_tp1

    def get_logps_with_carry(self, episode, carry):
        T, P = episode["act"].shape
        hiddens = self.observe_with_carry(episode, carry)['hs']
        hiddens = hiddens[:-1]  # remove last hidden state
        acts = episode["act"][:, self.player]
        logps = jax.vmap(self.logp)(hiddens, acts)
        assert logps.shape == (T,)
        return logps

    def logits(self, h):
        assert h.ndim == 1  # [features]
        logits = self.actor(h)
        return logits

    def get_logits_with_carry(self, episode, carry):
        T, P = episode["act"].shape
        hiddens = self.observe_with_carry(episode, carry)['hs']
        hiddens = hiddens[:-1]
        logits = jax.vmap(self.logits)(hiddens)
        assert logits.shape == (T, 4)
        return logits

    def emit(self, h, rng):
        assert h.ndim == 1  # [features]
        logits = self.logits(h)
        action = rax.categorical(rng, logits, axis=-1)
        return action

    def logps(self, h):
        assert h.ndim == 1  # [features]
        logits = self.logits(h)
        logps = jax.nn.log_softmax(logits, axis=-1)
        return logps

    def ps(self, h):
        assert h.ndim == 1  # [features]
        logits = self.logits(h)
        ps = jax.nn.softmax(logits, axis=-1)
        return ps

    def logp(self, h, action):
        logps = self.logps(h)
        logp = logps[action]
        return logp

    def value(self, h, t: int):
        assert h.ndim == 1  # [features]
        if self.use_film_in_value_for_time:
            x = self.film_t(h, jp.array([t]))
        else:
            x = jp.concatenate([h, jp.array([t])])  # add time feature
        value = self.value_function(x).reshape()  # changes shape from [1] to []
        return value

    def self_play_value(self, h, t: int):
        assert h.ndim == 1  # [features]
        if self.use_film_in_value_for_time:
            x = self.film_t_self_play(h, jp.array([t]))
        else:
            x = jp.concatenate([h, jp.array([t])])  # add time feature
        value = self.value_function_self_play(x).reshape()  # changes shape from [1] to []
        return value

    def get_initial_carry(self):
        assert self.aggregator_type.startswith('pola_gru'), "Only POLA GRU supports carry"
        return self.aggregator.get_initial_carry()

    def _init_dummy(self, episode):
        # flax initialization wraps around module methods only, not general code,
        # so we need a single method that involves all parameters :/
        hiddens = self.observe(episode)
        self.value(hiddens[0], 0)
        self.self_play_value(hiddens[0], 0)
        return jax.vmap(self.logits)(hiddens)


@struct.dataclass
class AgentParameters:
    params: "Any"
    module: "Module" = struct.field(pytree_node=False)

    @property
    def maybe_params(self):
        # RandomAgent doesn'tree have any, and flax doesn'tree represent this as an empty params dict
        return dict() if self.params is None else dict(params=self.params)

    @property
    def none_or_params(self):
        # so both TftAgent and ConvAgent can be used in the same code for mask
        return dict() if self.params is None else self.params

    @property
    def player(self):
        return self.module.player

    def observe(self, episode):
        return self.module.apply(self.maybe_params, episode, method=self.module.observe)

    def preprocess_player_obs(self, episode):
        return self.module.apply(self.maybe_params, episode, method=self.module.preprocess_player_obs)

    def emit(self, hiddens, rng):
        return self.module.apply(self.maybe_params, hiddens, rng, method=self.module.emit)

    def logp(self, hiddens, actions):
        return self.module.apply(self.maybe_params, hiddens, actions, method=self.module.logp)

    def logits(self, hiddens):
        return self.module.apply(self.maybe_params, hiddens, method=self.module.logits)

    def logps(self, hiddens):
        return self.module.apply(self.maybe_params, hiddens, method=self.module.logps)

    def ps(self, hiddens):
        return self.module.apply(self.maybe_params, hiddens, method=self.module.ps)

    def value(self, hidden, time):
        return self.module.apply(self.maybe_params, hidden, time, method=self.module.value)

    def self_play_value(self, hidden, time):
        return self.module.apply(self.maybe_params, hidden, time, method=self.module.self_play_value)

    def aggregate(self, hiddens):
        return self.module.apply(self.maybe_params, hiddens, method=self.module.aggregate)

    def observe_with_carry(self, episode, carry):
        return self.module.apply(self.maybe_params, episode, carry, method=self.module.observe_with_carry)

    def aggregate_with_carry(self, hiddens, carry):
        return self.module.apply(self.maybe_params, hiddens, carry, method=self.module.aggregate_with_carry)

    def get_single_action_with_carry(self, rng, episode, carry_0_t, t):
        return self.module.apply(self.maybe_params, rng, episode, carry_0_t, t, method=self.module.get_single_action_with_carry)

    def get_logps_with_carry(self, episode, carry):
        return self.module.apply(self.maybe_params, episode, carry, method=self.module.get_logps_with_carry)

    def get_initial_carry(self):
        return self.module.apply(self.maybe_params, method=self.module.get_initial_carry)


    def get_action(self, rng, episode, t, *args, **kwargs):
        hiddens = self.observe(episode)
        hiddens_t = jax.tree_map(lambda x: x[t], hiddens)
        action = self.emit(hiddens_t, rng)
        return action

    def get_logps(self, episode):
        T, P = episode["act"].shape
        hiddens = self.observe(episode)  # [1+T, P]
        hiddens = hiddens[:-1]  # remove last hidden state
        acts = episode["act"][:, self.player]
        logps = jax.vmap(self.logp)(hiddens, acts)
        assert logps.shape == (T,)
        return logps

    def get_logits(self, episode):
        hiddens = self.observe(episode)
        # [time, actions]
        logits = jax.vmap(self.logits)(hiddens)
        return logits

    def get_values(self, episode):
        hiddens = self.observe(episode)  # [trace_length+1]
        times = jp.arange(hiddens.shape[0])  # [trace_length+1]
        return jax.vmap(self.value)(hiddens, times)  # [trace_length+1]

    def get_self_play_values(self, episode):
        hiddens = self.observe(episode)  # [trace_length+1]
        times = jp.arange(hiddens.shape[0])  # [trace_length+1]
        return jax.vmap(self.self_play_value)(hiddens, times)  # [trace_length+1]

    def get_actor_mask(self):
        return ta.get_mask(self.none_or_params, ta.get_mask_for_key_func("actor"))

    def get_all_ones_mask(self):
        return ta.get_mask(self.none_or_params, ta.get_all_ones_mask_func())

    def __getattr__(self, name):
        if hasattr(self.module, name):
            method = getattr(self.module, name)

            def method_wrapper(*args, **kwargs):
                return self.module.apply(self.maybe_params, *args, **kwargs, method=method)

            return method_wrapper

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class ConvAggregator(nn.Module):
    activation: "Any"
    size: int
    window: int
    normalize: bool

    @nn.compact
    def __call__(self, x):
        # NOTE conv expects a batch axis, so None and squeeze it
        x = x[None]
        # NOTE left-padding makes it causal
        z = nn.Conv(features=self.size, kernel_size=(self.window,),
                    use_bias=not self.normalize, name="conv",
                    padding=[(self.window - 1, 0)])(x)
        z = z.squeeze(axis=0)
        if self.normalize:
            z = nn.LayerNorm(name="norm")(z)
        y = self.activation(z)
        return y


def init_agents(agent_modules, init_rng, coin_game, trace_length):
    episode = make_zero_episode(trace_length=trace_length, coin_game=coin_game)

    def init_func(i, am):
        return am.init(rscope(init_rng, str(i)), episode, method=am._init_dummy).get("params", None)

    paramss = [init_func(i, am) for i, am in enumerate(agent_modules)]
    agents = [AgentParameters(p, am) for p, am in zip(paramss, agent_modules)]
    return agents


class TftAgent(nn.Module):
    coin_game: "Any"
    grudge: bool  # always pick up opponent's coin if you picked up my coin even one time
    player: int

    def setup(self):
        assert self.coin_game.NUM_ACTIONS == 2  # because we can't observe space, we can't choose directional movements
        assert self.player == 0  # TftAgent implementation assumes it is player 0

    def observe(self, episode):
        ys = self.preprocess_player_obs(episode)
        return ys  # [time, cur_owner, xtook, otook]

    def preprocess_player_obs(self, episode):
        obs = episode["obs"][:, self.player]
        T = obs.shape[0]
        assert obs.shape[1:] == self.coin_game.OBS_SHAPE, f'obs.shape={obs.shape}, OBS_SHAPE={self.coin_game.OBS_SHAPE}'
        features = simplify_observations(obs, coin_game=self.coin_game, include_prev_owner=False)  # [time, owner, xtook, otook]
        features = features.reshape(T, -1)  # [time, owner*xtook*otook]

        if self.grudge:
            matrices = coin_taking_summary_matrix(player_obs=obs, coin_game_template=self.coin_game)  # [trace_length+1, num_players, num_players]
            should_grudge = matrices[:, 1, 0]  # number of coins taken by player 1 from player 0 (the tft agent)
            features = jp.concatenate([features, should_grudge.reshape(-1, 1)], axis=1)

        return features

    def aggregate(self, xs):
        return xs

    def logits(self, state):
        # state [current coin owner, i took, they took]
        # w shape [owner, xtook, otook]
        if self.grudge:
            assert state.shape == (9,)
            should_grudge = state[-1]
            state = state[:-1]

        assert state.shape == (8,)
        state = state.reshape(2, 2, 2)
        w = -np.ones([2, 2, 2], dtype="float32")
        w[0, :, :] = +1  # if it's my coin, always pick up
        w[1, :, 1] = +1  # if it's your coin, pick up iff you just picked up my coin last time
        w = 10 * w  # amplify logits
        sigmoid_logits = jp.einsum("ijk,ijk->", state, w)

        if self.grudge:
            sigmoid_logits = jp.where(should_grudge > 0, 10., sigmoid_logits)

        logits = jp.stack([jp.zeros_like(sigmoid_logits), sigmoid_logits], axis=-1)
        return logits

    def emit(self, hiddens, rng):
        logits = self.logits(hiddens)
        action = rax.categorical(rng, logits, axis=-1)
        return action

    def logps(self, h):
        logits = self.logits(h)
        logps = jax.nn.log_softmax(logits, axis=-1)
        return logps

    def logp(self, h, action):
        logps = self.logps(h)
        logp = logps[action]
        return logp

    def value(self, h, t: int):
        return 0.0

    def _init_dummy(self, episode):
        assert self.player == 0  # TftAgent implementation assumes it is player 0
        return None

def get_new_distances(episode, t, hp, rng, distance_of_this_player):
    other_player_pos = jax.lax.select(distance_of_this_player == 1, episode['player2_pos'][t], episode['player1_pos'][t])
    coin_pos = episode['coin_pos'][t]

    def get_new_distance(a):
        move = MOVES[a]
        new_agent_pos_x = (other_player_pos[0] + move[0]) % hp['height']
        new_agent_pos_y = (other_player_pos[1] + move[1]) % hp['width']
        new_dif_x = jp.abs(new_agent_pos_x - coin_pos[0])
        new_dif_y = jp.abs(new_agent_pos_y - coin_pos[1])
        wrapped_dif_x = jp.minimum(new_dif_x, hp['height'] - new_dif_x)
        wrapped_dif_y = jp.minimum(new_dif_y, hp['width'] - new_dif_y)
        return {'action': a, 'new_distance': wrapped_dif_x + wrapped_dif_y}

    new_distances = jax.vmap(lambda a: get_new_distance(a))(jp.arange(hp['g_num_actions']))
    shuffle_rng = rax.split(rng, 1)[0]
    # shuffle new_distances
    shuffled_distances = jax.tree_map(lambda x: rax.shuffle(shuffle_rng, x, axis=0), new_distances)
    # now sort by new distance
    indices = jp.argsort(shuffled_distances['new_distance'])
    sorted_distances = jax.tree_map(lambda x: x[indices], shuffled_distances)
    return {'new_distances': new_distances, 'sorted_distances': sorted_distances}


def get_cooperative_action(episode, t, hp, rng, agent_player, other_player):
    sorted_distances = get_new_distances(episode, t, hp, rng, distance_of_this_player=other_player)['sorted_distances']
    coin_owner = episode['coin_owner'][t]
    cooperative_action = jp.where(coin_owner == other_player, sorted_distances['action'][0], sorted_distances['action'][-1]).reshape()

    return cooperative_action

def get_defect_action(episode, t, hp, rng, agent_player, other_player):
    sorted_distances = get_new_distances(episode, t, hp, rng, distance_of_this_player=other_player)['sorted_distances']
    coin_owner = episode['coin_owner'][t]
    defect_action = jp.where(coin_owner == other_player, sorted_distances['action'][0], sorted_distances['action'][0]).reshape()

    return defect_action


def test_replay_buffer_agent():
    game_template = CoinGame(rng=jax.random.PRNGKey(0),
                             HEIGHT=2,
                             WIDTH=2,
                             NUM_ACTIONS=4,
                             coin_owner=0,
                             coin_pos=jp.array([0, 0]),
                             players_pos=jp.array([[0, 1], [1, 1]]),
                             ACTION_NAMES=tuple("left right up down".split()),
                             OBS_SHAPE=(4, 2, 2),
                             trace_length=16,
                             new_coin_every_turn=False,
                             )

    config = {'mode': 'custom',
              'use_takers_summary_matrix_features': True,
              'normalize_takers_summary_matrix': False,
              'use_manual_features': False,
              'use_reward_features': False,
              'use_diffret_features': False,
              'use_action_features': False,
              }

    agent_impl = ConvAgent(coin_game=game_template,
                           conv_aggregator_activation='relu',
                           preprocess_obs_config=config,
                           value_mlp_features=[1],
                           actor_mlp_features=[4],
                           horizon=-1,
                           use_film_in_value_for_time=0,
                           film_size=32,
                           aggregator_mlp_features=[32, 32],
                           aggregator_type='mlp-residual-fixed',
                           player=0,
                           )
    agent = init_agents([agent_impl], init_rng=rax.PRNGKey(0), coin_game=game_template, trace_length=16)[0]

    agent_rb = [agent.params for _ in range(16)]
    # concatenate all params
    agent_rb = jax.tree_map(lambda *args: jp.concatenate(args, axis=0), *agent_rb)
    # sample 8 params
    agent_indices = jax.random.randint(rax.PRNGKey(0), shape=(8,), minval=0, maxval=16)
    agent_rb = jax.tree_map(lambda x: x[agent_indices], agent_rb)
    # print agent_rb structure
    print(jax.tree_map(lambda x: x.shape, agent_rb))



if __name__ == '__main__':
    test_replay_buffer_agent()
