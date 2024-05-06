import dataclasses
import time
from functools import partial
from typing import Tuple

import jax
import numpy as np
import tabulate
from flax import struct as struct
from jax import random as rax, numpy as jp
import flax.linen as nn

from league._utils import rscope, eps

MOVES = jp.array([[0, -1], [0, +1], [-1, 0], [+1, 0]])

# taken from https://github.com/google/flax/issues/3032
def dataclass_eq_return_false(cls):
  def eq(a, b):
    return False
  cls = struct.dataclass(cls)
  cls.__eq__ = eq
  return cls

# NOTE difference from LOLA paper coin game: players/coins may spawn on
# top of each other
@dataclass_eq_return_false
class CoinGame:
    rng: "PRNGKey"
    coin_owner: "int[]"
    coin_pos: "int[2]"
    players_pos: "int[NUM_PLAYERS]"
    trace_length: int = struct.field(pytree_node=False)
    new_coin_every_turn: bool = struct.field(pytree_node=False)
    prev_coin_takers: "int[2]"  # prev_coin_takers[i] is the player who took the last coin of color i

    HEIGHT: int = struct.field(pytree_node=False)
    WIDTH: int = struct.field(pytree_node=False)
    ACTION_NAMES: "Any" = struct.field(pytree_node=False)
    NUM_ACTIONS: int = struct.field(pytree_node=False)  # 2 for take/leave or 4 for left/right/up/down
    OBS_SHAPE: Tuple[int, int, int] = struct.field(pytree_node=False)
    NUM_PLAYERS = 2
    SYMBOLS = tuple("XOxo")
    dtype: str = struct.field(pytree_node=False, default="float32")
    # sample coin position so that it does not coincide with any player (like LOLA/POLA)
    # NOTE only works if the board can accommodate it
    COIN_MAY_LAND_ON_PLAYER = False

    @classmethod
    def init(cls, rng, height, width, gnumactions, trace_length, new_coin_every_turn, dtype):
        rng, owner_rng, position_rng = rax.split(rng, 3)
        coin_owner = rax.randint(owner_rng, minval=0, maxval=2, shape=[])
        players_pos = jax.vmap(lambda r: cls._random_position(r, height, width))(rax.split(position_rng, 2))
        coin_pos = cls._random_coin_position(rscope(position_rng, "coin"), players_pos, height, width)
        assert height * width >= 2, "board too small for the coin not to land on players"

        if gnumactions == 2:
            action_names = tuple("leave take".split())
        else:
            action_names = tuple("left right up down".split())

        env = cls(rng=rng,
                  coin_owner=coin_owner,
                  coin_pos=coin_pos,
                  players_pos=players_pos,
                  OBS_SHAPE=(2 * cls.NUM_PLAYERS, height, width),
                  HEIGHT=height,
                  WIDTH=width,
                  NUM_ACTIONS=gnumactions,
                  ACTION_NAMES=action_names,
                  trace_length=trace_length,
                  new_coin_every_turn=new_coin_every_turn,
                  prev_coin_takers=jp.array([-1, -1.]),
                  dtype=dtype)

        obs = env.get_obs()
        return env, obs

    def step(self, actions):
        # takers get 1, coin_owner gets -2 if any other taker
        assert actions.shape == (self.NUM_PLAYERS,)
        if self.NUM_ACTIONS == 4:
            # NOTE does not work with LogitsAgent (it doesn't get space info, so can'tree reasonably emit left/right/up/down)
            moves = MOVES[actions]
            new_players_pos = (self.players_pos + moves) % jp.array([self.HEIGHT, self.WIDTH])[None]
            takers = (new_players_pos == self.coin_pos[None]).all(axis=1)
        else:  # action 1 is pickup
            takers = actions == 1  # [player]
            # put takers into the coin cell, nontakers into the other cell. that way agent_loaders can see
            # who picked up the coin.
            assert self.HEIGHT == 1 and self.WIDTH == 2
            noncoinpos = jp.array([self.coin_pos[0], 1 - self.coin_pos[1]], dtype=self.coin_pos.dtype)
            new_players_pos = jp.where(takers[:, None], self.coin_pos[None, :], noncoinpos[None, :])

        assert new_players_pos.shape == self.players_pos.shape
        owner = jp.eye(self.NUM_PLAYERS, dtype="bool")[self.coin_owner]
        rewards = 1 * takers - 2 * owner * (takers & ~owner).any()

        last_taker_coin_0, last_taker_coin_1 = self.prev_coin_takers[0], self.prev_coin_takers[1]
        last_taker_coin_0 = jp.where((self.coin_owner == 0) & takers[0], 0, last_taker_coin_0)
        last_taker_coin_0 = jp.where((self.coin_owner == 0) & takers[1], 1, last_taker_coin_0)
        last_taker_coin_1 = jp.where((self.coin_owner == 1) & takers[1], 1, last_taker_coin_1)
        last_taker_coin_1 = jp.where((self.coin_owner == 1) & takers[0], 0, last_taker_coin_1)

        new_rng, rng = rax.split(self.rng)
        new_coin_pos = self._random_coin_position(new_rng, new_players_pos, self.HEIGHT, self.WIDTH)
        new_coin_owner = (self.coin_owner + 1) % self.NUM_PLAYERS

        new_rng, rng = rax.split(self.rng)

        if not self.new_coin_every_turn:
            # update coin pos/owner only if it was taken
            new_coin_pos = jp.where(takers.any(), new_coin_pos, self.coin_pos)
            new_coin_owner = jp.where(takers.any(), new_coin_owner, self.coin_owner)

        env = self.replace(
            rng=new_rng,
            players_pos=new_players_pos,
            coin_pos=new_coin_pos,
            coin_owner=new_coin_owner,
            prev_coin_takers=jp.stack([last_taker_coin_0, last_taker_coin_1], axis=0)
        )
        obs = env.get_obs()
        return env, obs, rewards

    @classmethod
    def _random_position(cls, rng, height, width):
        pos = rax.randint(rng, minval=0, maxval=height * width, shape=[])
        return jp.array([pos // width, pos % width])

    @classmethod
    def _random_coin_position(cls, rng, players_pos, height, width):
        if cls.COIN_MAY_LAND_ON_PLAYER:
            return cls._random_position(rng)
        # like LOLA/POLA, sample coin so that it is never in the same place as any player
        assert height * width > cls.NUM_PLAYERS  # only possible if there is room
        players_pos_flat = players_pos[:, 0] * width + players_pos[:, 1]
        occupied = (players_pos_flat[:, None] == jp.arange(height * width)[None, :]).any(axis=0)
        coin_pos_flat = rax.choice(rng, occupied.size, p=1 - occupied, shape=[])
        coin_pos = jp.stack([coin_pos_flat // width, coin_pos_flat % width])
        return coin_pos

    def get_obs(self):
        x = jp.zeros(self.OBS_SHAPE, dtype=self.dtype)
        x = x.at[jp.arange(self.NUM_PLAYERS), self.players_pos[:, 0], self.players_pos[:, 1]].set(1.)
        x = x.at[self.NUM_PLAYERS + self.coin_owner, self.coin_pos[0], self.coin_pos[1]].set(1.)
        # show the second player the state as if they were the first player, so policies are swappable
        xT = x[((1, 0, 3, 2),)]  # swap channels (X, O, x, o) -> (O, X, o, x)
        return jp.stack([x, xT])

    def renderframe(self, obs, act, rew):
        if self.NUM_ACTIONS == 2:
            lines = []
            for p, s in enumerate("xo"):
                parts = [s if act[p] else " ",
                         f"{int(rew[p]):+}"]
                lines.append("".join(parts))
            return "\n".join(lines)
        else:
            P, C, I, J = obs.shape
            x = np.zeros((I, J), dtype="str")
            x.fill("-")
            locations = dict()
            for [c, i, j] in np.argwhere(obs[0]):
                x[i, j] = self.SYMBOLS[c]
                locations[self.SYMBOLS[c]] = (i, j)
            lines = []
            lines.extend((": " + "".join(row) + " :") for row in x)
            lines.extend(f"{s} @ {i},{j}" for s, (i, j) in locations.items())
            for p in range(P):
                s = self.SYMBOLS[p]
                a = self.ACTION_NAMES[act[p]]
                lines.append(f"{s}: {a} {int(rew[p])}")
            return "\n".join(lines)

    def copy(self):
        return self.replace()

    def get_taker_summary_matrix_features(self, obs, normalize, horizon):
        T = obs.shape[0]  # note: T = trace_length + 1
        matrices = coin_taking_summary_matrix(player_obs=obs, coin_game_template=self)  # [trace_length+1, num_players, num_players]
        matrices = matrices.reshape([T, -1])  # [trace_length+1, num_players * num_players]
        if normalize:
            matrices = matrices / (jp.concatenate((jp.ones(1, obs.dtype), jp.arange(matrices.shape[0] - 1) + 1)))[..., None]
        if horizon != -1:
            shifted_matrices = jp.concatenate((jp.zeros((horizon, matrices.shape[1]), dtype=obs.dtype), matrices[:-horizon, :]))
            matrices = matrices - shifted_matrices

        return matrices

    def get_manual_features(self, obs):
        # agent should be able to infer act from obs, but it struggles to learn.
        # manually extract TFT-relevant features and add them in.
        T = obs.shape[0]
        features = simplify_observations(obs, coin_game=self.coin_game)  # [time, owner, xtook, otook]
        features = features.reshape([T, -1])  # [time, owner * xtook * otook]
        return features

    def get_rew_features(self, episode):
        head_zero_rew = jp.concatenate([jp.zeros((1, 2)), episode["rew"]], axis=0)
        return head_zero_rew

    def get_diffret_features(self, episode):
        head_zero_rew = jp.concatenate([jp.zeros((1, 2)), episode["rew"]], axis=0)
        ret = head_zero_rew.cumsum(axis=1)
        return (ret[:, 0] - ret[:, 1])[:, None]

    def get_action_features(self, player, episode):
        T = episode["obs"].shape[0]
        act = jp.where(player == 0, episode["act"], episode["act"][:, ::-1])
        A = self.coin_game.NUM_ACTIONS
        act = jp.eye(A, dtype="float32")[act]  # [time, player, action]
        act = jp.concatenate([jp.zeros_like(act[:1]), act], axis=0)
        return act.reshape([T, -1])

    def get_juan_bits(self, episode, player):
        def change_ownership(x):
            # change 0s to 1s and 1s to 0s and ignore -1s
            old_x = x
            x = jp.where(old_x == 0, 1, x)
            x = jp.where(old_x == 1, 0, x)
            return x

        player_coin_last_takers = jp.where(player == 0, episode['games'].prev_coin_takers[:, 0], change_ownership(episode['games'].prev_coin_takers[:, 1]))
        other_coin_last_takers = jp.where(player == 0, episode['games'].prev_coin_takers[:, 1], change_ownership(episode['games'].prev_coin_takers[:, 0]))
        # add a dimension to end of each
        player_coin_last_takers = jp.expand_dims(player_coin_last_takers, axis=1)
        other_coin_last_takers = jp.expand_dims(other_coin_last_takers, axis=1)
        return player_coin_last_takers, other_coin_last_takers

    def old_get_juan_bits(self, player_obs):
        taker_matrices = jax.vmap(lambda prev, cur: matrix_for_timestep(prev_xs=prev, xs=cur, coin_game_template=self))(player_obs[:-1], player_obs[1:])
        zero_frame = jp.zeros((1, self.NUM_PLAYERS, self.NUM_PLAYERS))
        taker_matrices = jp.concatenate([zero_frame, taker_matrices], axis=0)

        player = 0
        other = 1

        def last_taker(carry, taker_matrix_at_t):
            last_taker_player_coin, last_taker_other_coin = carry
            last_taker_player_coin = jp.where(taker_matrix_at_t[player, player] == 1, player, last_taker_player_coin)
            last_taker_player_coin = jp.where(taker_matrix_at_t[other, player] == 1, other, last_taker_player_coin)
            last_taker_other_coin = jp.where(taker_matrix_at_t[other, other] == 1, other, last_taker_other_coin)
            last_taker_other_coin = jp.where(taker_matrix_at_t[player, other] == 1, player, last_taker_other_coin)
            return (last_taker_player_coin, last_taker_other_coin), (last_taker_player_coin, last_taker_other_coin)

        _, (player_coin_last_takers, other_coin_last_takers) = jax.lax.scan(f=last_taker, init=(-1, -1), xs=taker_matrices)

        player_coin_last_takers = player_coin_last_takers[:, None]  # [trace_length+1, 1]
        other_coin_last_takers = other_coin_last_takers[:, None]  # [trace_length+1, 1]

        return player_coin_last_takers, other_coin_last_takers

    def preprocess_player_obs(self,
                              episode,
                              player: int,
                              horizon: int,
                              config: dict):

        mode = config["mode"]
        obs = episode["obs"][:, player]
        assert obs.shape[1:] == self.OBS_SHAPE, f'obs.shape={obs.shape}, OBS_SHAPE={self.OBS_SHAPE}'
        T = obs.shape[0]  # note: T = trace_length + 1

        if mode == "raw":
            return obs

        elif mode == "raw_flat":
            return obs.reshape([T, -1])

        elif mode == "minimal":
            tsm_features = self.get_taker_summary_matrix_features(obs, normalize=False, horizon=horizon)  # [trace_length+1, num_players * num_players]
            # TODO we should permute tsm_features if player == 1, so agent can play as either player in self-play
            is_coin_mine = (episode['coin_owner'] == player)  # [trace_length+1, 1]
            my_position = jp.where(player == 0, episode["player1_pos"], episode["player2_pos"])  # [trace_length+1, 2]
            other_position = jp.where(player == 0, episode["player2_pos"], episode["player1_pos"])  # [trace_length+1, 2]
            coin_position = episode["coin_pos"]  # [trace_length+1, 2]
            x = jp.concatenate([tsm_features, is_coin_mine, coin_position - my_position, coin_position - other_position], axis=1)
            return x

        elif mode == "juan":
            player_coin_last_takers, other_coin_last_takers = self.get_juan_bits(episode, player)
            is_coin_mine = (episode['coin_owner'] == player)  # [trace_length+1, 1]
            my_position = jp.where(player == 0, episode["player1_pos"], episode["player2_pos"])  # [trace_length+1, 2]
            other_position = jp.where(player == 0, episode["player2_pos"], episode["player1_pos"])  # [trace_length+1, 2]
            coin_position = episode["coin_pos"]  # [trace_length+1, 2]
            xs = jp.concatenate([player_coin_last_takers, other_coin_last_takers, is_coin_mine, coin_position - my_position, coin_position - other_position], axis=1)
            return xs

        elif mode == 'old_juan':
            old_player_coin_last_takers, old_other_coin_last_takers = self.old_get_juan_bits(obs)
            is_coin_mine = (episode['coin_owner'] == player)  # [trace_length+1, 1]
            my_position = jp.where(player == 0, episode["player1_pos"], episode["player2_pos"])  # [trace_length+1, 2]
            other_position = jp.where(player == 0, episode["player2_pos"], episode["player1_pos"])  # [trace_length+1, 2]
            coin_position = episode["coin_pos"]  # [trace_length+1, 2]
            xs = jp.concatenate([old_player_coin_last_takers, old_other_coin_last_takers, is_coin_mine, coin_position - my_position, coin_position - other_position], axis=1)
            return xs

        elif mode == "juan_and_raw_obs":
            x = obs.reshape([T, -1])
            player_coin_last_takers, other_coin_last_takers = self.get_juan_bits(episode, player)
            x = jp.concatenate([player_coin_last_takers, other_coin_last_takers, x], axis=1)
            return x

        elif mode == "custom":
            x = obs.reshape([T, -1])

            if config['use_takers_summary_matrix_features']:
                if horizon != -1:
                    assert ~(config['use_manual_features'] | config['use_reward_features'] | config['use_diffret_features'] | config['use_action_features'])
                tsm_features = self.get_taker_summary_matrix_features(obs=obs, normalize=config['normalize_takers_summary_matrix'], horizon=horizon)
                x = jp.concatenate([tsm_features, x], axis=1)

            if config['use_manual_features']:
                manual_features = self.get_manual_features(obs=obs)
                x = jp.concatenate([manual_features, x], axis=1)

            if config['use_reward_features']:
                rew_features = self.get_rew_features(episode=episode)
                x = jp.concatenate(rew_features, x, axis=1)

            if config['use_diffret_features']:
                dr_features = self.get_diffret_features(episode=episode)
                x = jp.concatenate([dr_features, x], axis=1)

            if config['use_action_features']:
                act_features = self.get_action_features(player=player, episode=episode)
                x = jp.concatenate([x, act_features], axis=1)
            return x


        else:
            raise NotImplementedError(f"mode={mode} not implemented.")

    def distance_to_coin(self):
        return jp.abs(self.players_pos - self.coin_pos).sum(axis=-1)

    def distance_to_cell(self, cell):
        return jp.abs(self.players_pos - cell).sum(axis=-1)

    def printepisode(self, episodes, batch_max=10, time_max=32):
        obs = episodes["obs"]
        act = episodes["act"]
        rew = episodes["rew"]
        B, T, P = rew.shape
        rows = []
        for b in range(B)[:batch_max]:
            frames = []
            for t in range(T)[:time_max]:
                frame = self.renderframe(obs[b, t], act[b, t], rew[b, t])
                frames.append(frame)
            rows.append(frames)
        tabulate.PRESERVE_WHITESPACE = True
        return tabulate.tabulate(rows, tablefmt="grid")


class SmoothObs(nn.Module):
    game_template: CoinGame

    @nn.compact
    def __call__(self, xs):
        def transform_to_minimal_obs(x):
            assert x.shape == (self.game_template.NUM_PLAYERS * self.game_template.NUM_PLAYERS + 5,)
            T = self.game_template.trace_length
            P = self.game_template.NUM_PLAYERS
            H = self.game_template.HEIGHT
            W = self.game_template.WIDTH
            tsm_features = nn.sigmoid(x[:P * P]) * T
            is_coin_mine = nn.sigmoid(x[P * P:P * P + 1])
            diff_mine_x = nn.tanh(x[P * P + 1:P * P + 2]) * W
            diff_mine_y = nn.tanh(x[P * P + 2:P * P + 3]) * H
            diff_other_x = nn.tanh(x[P * P + 3:P * P + 4]) * W
            diff_other_y = nn.tanh(x[P * P + 4:P * P + 5]) * H
            return jp.concatenate([tsm_features, is_coin_mine, diff_mine_x, diff_mine_y, diff_other_x, diff_other_y], axis=0)

        return jax.vmap(transform_to_minimal_obs)(xs)


def test_smooth_obs():
    game_template = CoinGame(rng=jax.random.PRNGKey(0),
                             HEIGHT=2,
                             WIDTH=2,
                             NUM_ACTIONS=4,
                             coin_owner=0,
                             coin_pos=jp.array([0, 0]),
                             players_pos=jp.array([[0, 1], [1, 1]]),
                             ACTION_NAMES=tuple("left right up down".split()),
                             OBS_SHAPE=(4, 2, 2),
                             trace_length=8,
                             new_coin_every_turn=False,
                             )
    smooth_obs = SmoothObs(game_template=game_template)
    smooth_obs_params = smooth_obs.init(jax.random.PRNGKey(42), jp.ones((1, 9)))
    obs = smooth_obs.apply(smooth_obs_params, np.random.randn(64, 9))
    print(obs.shape, 'obs.shape')
    print(obs, 'obs')

def matrix_for_timestep(prev_xs, xs, coin_game_template):
    assert xs.shape == (2 * coin_game_template.NUM_PLAYERS, coin_game_template.HEIGHT, coin_game_template.WIDTH)
    P = coin_game_template.NUM_PLAYERS
    prev_coin_mask = prev_xs[P:].sum(axis=0)  # [H, W] prev_coin_mask[i,j] = 1 iff there was a coin at (i,j) in the previous timestep
    curr_player_masks = xs[:P]  # [P, H, W] curr_player[i][h][w] is 1 if player i is at (h,w)
    prev_takers = (curr_player_masks * prev_coin_mask[None]).sum(axis=(1, 2))  # [P] prev_takers[p] is in [0,1], 1 indicates player p took the coin at the previous timestep
    prev_owner = prev_xs[P:].sum(axis=(1, 2))  # [P] prev_owner[p] is in [0,1], 1 indicates player p owned the coin at the previous timestep
    taking_matrix = jp.einsum('t,o->to', prev_takers, prev_owner)  # [P, P] taking_matrix[i,j] is the number of coins player i took from j-colored coins
    return taking_matrix

@jax.jit
def coin_taking_summary_matrix(player_obs, coin_game_template):
    """convert observation of the player [trace_length+1, 2*num_players channels, *board space]
       to a summary matrix [trace_length+1, 2*num_players, 2*num_players]
       where the (i,j)th entry is the number of coins the ith player has taken from the j-colored coins
    """

    # prepend all-zero frame for the initial state
    zero_frame = jp.zeros([1, *player_obs.shape[1:]], dtype=player_obs.dtype)  # [1, 2*num_players, height, width]
    prepended_obs = jp.concatenate([zero_frame, player_obs], axis=0)  # [trace_length+2, 2*num_players, height, width]
    previous_time_step_obs = prepended_obs[:-1]  # [trace_length+1, 2*num_players, height, width]
    cur_time_step_obs = prepended_obs[1:]  # [trace_length+1, 2*num_players, height, width]
    matrices = jax.vmap(lambda a, b: matrix_for_timestep(a, b, coin_game_template))(previous_time_step_obs, cur_time_step_obs)  # [time, P, P]
    matrices = jp.cumsum(matrices, axis=0)  # [trace_length+1, P, P] matrices[tree][i][j] is the number of coins player i has taken from j-colored coins at the end of timestep tree
    return matrices


def simplify_observations(xs, coin_game, include_prev_owner: bool = True):
    # convert xs [time, channels, *space]
    # to simplified representation
    # [time, curr_owner, x_took, o_took]
    # that is one-hot.
    @jax.vmap  # across time
    def within_time(prev_xs, xs):
        assert xs.shape == (2 * coin_game.NUM_PLAYERS, coin_game.HEIGHT, coin_game.WIDTH)
        P = coin_game.NUM_PLAYERS
        # turn the 2-history of spatial information into three bits:
        # who is the current owner, and, for both players, whether they picked up the previous coin
        # continuous relaxation (diffable so works with ProbeVHat)
        prev_coin_mask = prev_xs[P:].sum(axis=0)  # across coin channels (one-hot)
        curr_player_masks = xs[:P]
        # prev_takers[p] is in [0,1], 1 indicates player p took
        prev_takers = (curr_player_masks * prev_coin_mask[None]).sum(axis=(1, 2))  # across space
        prev_xtook = jp.array([1 - prev_takers[0], prev_takers[0]])
        prev_otook = jp.array([1 - prev_takers[1], prev_takers[1]])
        curr_owner = xs[P:].sum(axis=(1, 2))  # one hot
        prev_owner = prev_xs[P:].sum(axis=(1, 2))  # one hot
        # > 0.1 is just there because prev_xs is a float and we want to avoid numerical issues in casting to bool
        prev_owner = jp.concatenate([jp.array([1 - (prev_xs[P:] > 0.1).any()]), prev_owner])

        if include_prev_owner:
            state = jp.einsum("a,b,c,d->abcd", prev_owner, curr_owner, prev_xtook, prev_otook)
        else:
            state = jp.einsum("a,b,c->abc", curr_owner, prev_xtook, prev_otook)
        return state

    # prepend all-zero frame for the initial state
    xs = jp.concatenate([jp.zeros([1, *xs.shape[1:]], dtype=xs.dtype), xs], axis=0)
    state = within_time(xs[:-1], xs[1:])  # [time, bits...]
    return state


def distance_change_after_step(coin_game, actions):
    coin_pos = coin_game.coin_pos
    start_distances = coin_game.distance_to_coin()
    new_coin_game, obs, rew = coin_game.step(actions)
    finish_distances = new_coin_game.distance_to_cell(coin_pos)
    return finish_distances - start_distances


def test_distance_function():
    game = CoinGame(rng=jax.random.PRNGKey(0),
                    HEIGHT=2,
                    WIDTH=2,
                    NUM_ACTIONS=4,
                    coin_owner=0,
                    coin_pos=jp.array([0, 0]),
                    players_pos=jp.array([[0, 1], [1, 1]]),
                    ACTION_NAMES=tuple("left right up down".split()),
                    OBS_SHAPE=(4, 2, 2),
                    trace_length=8,
                    new_coin_every_turn=True,
                    )

    # test that the distance function is correct
    distance_to_coin = game.distance_to_coin()
    print(f'distance_to_coin: {distance_to_coin}')
    assert jp.allclose(distance_to_coin, jp.array([1, 2]))
    actions = jp.array([0, 2])
    distance_change = distance_change_after_step(game, actions)
    print(f'distance_change: {distance_change}')
    assert jp.allclose(distance_change, jp.array([-1, -1]))


def test_play_episode():
    game = CoinGame(rng=jax.random.PRNGKey(0),
                    HEIGHT=2,
                    WIDTH=2,
                    NUM_ACTIONS=4,
                    coin_owner=0,
                    coin_pos=jp.array([0, 0]),
                    players_pos=jp.array([[0, 1], [1, 1]]),
                    ACTION_NAMES=tuple("left right up down".split()),
                    OBS_SHAPE=(4, 2, 2),
                    trace_length=8,
                    new_coin_every_turn=False,
                    )

    def get_actions(subrng, env, episode, t):
        return jp.array([0, 2]), None

    episode, _ = play_episode_scan(env=game, get_actions=get_actions, rng=jax.random.PRNGKey(0), trace_length=8)
    return


def make_zero_episode(trace_length, coin_game, dtype='float32'):
    # get something with the same structure as play_episode would return
    obs = jp.zeros([1 + trace_length, 2, *coin_game.OBS_SHAPE], dtype=dtype)
    act = jp.zeros([trace_length, 2], dtype="int32")
    rew = jp.zeros([trace_length, 2], dtype=dtype)
    coin_pos = jp.zeros([trace_length + 1, 2], dtype="int32")
    coin_owner = jp.zeros([trace_length + 1, 1], dtype="int32")
    player1_pos = jp.zeros([trace_length + 1, 2], dtype="int32")
    player2_pos = jp.zeros([trace_length + 1, 2], dtype="int32")
    games = jax.tree_map(lambda x: jp.expand_dims(x, axis=0).repeat(trace_length+1, axis=0), coin_game)
    return dict(obs=obs,
                act=act,
                rew=rew,
                coin_pos=coin_pos,
                coin_owner=coin_owner,
                player1_pos=player1_pos,
                player2_pos=player2_pos,
                games=games)


def play_episode_unroll(env, get_actions, rng, trace_length, qa_auxes=None):
    episode = make_zero_episode(trace_length=trace_length, coin_game=env)

    # set initial observations
    episode["obs"] = episode["obs"].at[0].set(env.get_obs())
    episode["coin_pos"] = episode["coin_pos"].at[0].set(env.coin_pos)
    episode["coin_owner"] = episode["coin_owner"].at[0].set(env.coin_owner)
    episode["player1_pos"] = episode["player1_pos"].at[0].set(env.players_pos[0])
    episode["player2_pos"] = episode["player2_pos"].at[0].set(env.players_pos[1])
    start_time = time.time()
    auxes = []
    for t in range(trace_length):
        qa_aux = jax.tree_map(lambda x: x[t], qa_auxes) if qa_auxes is not None else None
        rng, subrng = rax.split(rng)
        old_game = env
        episode['games'] = jax.tree_map(lambda x, o: x.at[t].set(o), episode['games'], old_game)
        act, aux = get_actions(subrng, env, episode, t) if qa_aux is None else get_actions(subrng, env, episode, t, qa_aux=qa_aux)
        env, obs, rew = env.step(act)
        jax.debug.print(f't: {t} time passed: {time.time() - start_time}')
        episode["obs"] = episode["obs"].at[1 + t].set(obs)
        episode["coin_pos"] = episode["coin_pos"].at[1 + t].set(env.coin_pos)
        episode["coin_owner"] = episode["coin_owner"].at[1 + t].set(env.coin_owner)
        episode["player1_pos"] = episode["player1_pos"].at[1 + t].set(env.players_pos[0])
        episode["player2_pos"] = episode["player2_pos"].at[1 + t].set(env.players_pos[1])
        episode["act"] = episode["act"].at[t].set(act)
        episode["rew"] = episode["rew"].at[t].set(rew)
        auxes.append(aux)

    last_game = env
    episode['games'] = jax.tree_map(lambda x, o: x.at[trace_length].set(o), episode['games'], last_game)

    auxes = jax.tree_util.tree_map(lambda *xs: jp.stack(xs, axis=0), *auxes)
    return episode, auxes


def play_episode_scan(env, get_actions, rng, trace_length, qa_auxes=None):
    episode = make_zero_episode(trace_length=trace_length, coin_game=env, dtype=env.dtype)

    # set initial observations
    episode["obs"] = episode["obs"].at[0].set(env.get_obs())
    episode["coin_pos"] = episode["coin_pos"].at[0].set(env.coin_pos)
    episode["coin_owner"] = episode["coin_owner"].at[0].set(env.coin_owner)
    episode["player1_pos"] = episode["player1_pos"].at[0].set(env.players_pos[0])
    episode["player2_pos"] = episode["player2_pos"].at[0].set(env.players_pos[1])

    def body_fn(carry, _):
        env, rng, episode, t, qa_auxes = carry
        qa_aux = jax.tree_map(lambda x: x[t], qa_auxes) if qa_auxes is not None else None
        rng, subrng = rax.split(rng)
        old_game = env
        episode['games'] = jax.tree_map(lambda x, o: x.at[t].set(o), episode['games'], env)
        act, aux = get_actions(subrng, env, episode, t) if qa_aux is None else get_actions(subrng, env, episode, t, qa_aux=qa_aux)
        assert act.shape == (2,)
        env, obs, rew = env.step(act)
        episode["obs"] = episode["obs"].at[1 + t].set(obs)
        episode["coin_pos"] = episode["coin_pos"].at[1 + t].set(env.coin_pos)
        episode["coin_owner"] = episode["coin_owner"].at[1 + t].set(env.coin_owner)
        episode["player1_pos"] = episode["player1_pos"].at[1 + t].set(env.players_pos[0])
        episode["player2_pos"] = episode["player2_pos"].at[1 + t].set(env.players_pos[1])
        episode["act"] = episode["act"].at[t].set(act)
        episode["rew"] = episode["rew"].at[t].set(rew)
        return (env, rng, episode, t + 1, qa_auxes), aux

    (env, rng, episode, _, _), aux = jax.lax.scan(f=body_fn, init=(env, rng, episode, 0, qa_auxes), xs=(), length=trace_length)
    last_game = env
    episode['games'] = jax.tree_map(lambda x, o: x.at[trace_length].set(o), episode['games'], last_game)
    return episode, aux

def play_episode_scan_inner_gru(env, get_actions, rng, trace_length, qa_auxes=None, agent_carry_0_t=None):
    episode = make_zero_episode(trace_length=trace_length, coin_game=env, dtype=env.dtype)

    # set initial observations
    episode["obs"] = episode["obs"].at[0].set(env.get_obs())
    episode["coin_pos"] = episode["coin_pos"].at[0].set(env.coin_pos)
    episode["coin_owner"] = episode["coin_owner"].at[0].set(env.coin_owner)
    episode["player1_pos"] = episode["player1_pos"].at[0].set(env.players_pos[0])
    episode["player2_pos"] = episode["player2_pos"].at[0].set(env.players_pos[1])

    def body_fn(carry, _):
        env, rng, episode, t, qa_auxes, agent_carry_0_t = carry
        qa_aux = jax.tree_map(lambda x: x[t], qa_auxes) if qa_auxes is not None else None
        rng, subrng = rax.split(rng)
        old_game = env
        episode['games'] = jax.tree_map(lambda x, o: x.at[t].set(o), episode['games'], env)
        act, aux = get_actions(subrng, env, episode, t, agent_carry_0_t) if qa_aux is None else get_actions(subrng, env, episode, t, agent_carry_0_t, qa_aux=qa_aux)
        agent_carry_0_tp1 = aux['agent_carry_0_tp1']
        assert act.shape == (2,)
        env, obs, rew = env.step(act)
        episode["obs"] = episode["obs"].at[1 + t].set(obs)
        episode["coin_pos"] = episode["coin_pos"].at[1 + t].set(env.coin_pos)
        episode["coin_owner"] = episode["coin_owner"].at[1 + t].set(env.coin_owner)
        episode["player1_pos"] = episode["player1_pos"].at[1 + t].set(env.players_pos[0])
        episode["player2_pos"] = episode["player2_pos"].at[1 + t].set(env.players_pos[1])
        episode["act"] = episode["act"].at[t].set(act)
        episode["rew"] = episode["rew"].at[t].set(rew)
        return (env, rng, episode, t + 1, qa_auxes, agent_carry_0_tp1), aux

    (env, rng, episode, _, _, _), aux = jax.lax.scan(f=body_fn, init=(env, rng, episode, 0, qa_auxes, agent_carry_0_t), xs=(), length=trace_length)
    last_game = env
    episode['games'] = jax.tree_map(lambda x, o: x.at[trace_length].set(o), episode['games'], last_game)
    return episode, aux

def play_episode_scan_agent_detective_gru(env, get_actions, rng, trace_length, qa_auxes=None, agent_carry_0_t=None, detective_carry_0_t=None):
    episode = make_zero_episode(trace_length=trace_length, coin_game=env, dtype=env.dtype)

    # set initial observations
    episode["obs"] = episode["obs"].at[0].set(env.get_obs())
    episode["coin_pos"] = episode["coin_pos"].at[0].set(env.coin_pos)
    episode["coin_owner"] = episode["coin_owner"].at[0].set(env.coin_owner)
    episode["player1_pos"] = episode["player1_pos"].at[0].set(env.players_pos[0])
    episode["player2_pos"] = episode["player2_pos"].at[0].set(env.players_pos[1])

    def body_fn(carry, _):
        env, rng, episode, t, qa_auxes, agent_carry_0_t, detective_carry_0_t = carry
        qa_aux = jax.tree_map(lambda x: x[t], qa_auxes) if qa_auxes is not None else None
        rng, subrng = rax.split(rng)
        old_game = env
        episode['games'] = jax.tree_map(lambda x, o: x.at[t].set(o), episode['games'], env)
        act, aux = get_actions(subrng,
                               env,
                               episode,
                               t,
                               agent_carry_0_t=agent_carry_0_t,
                               detective_carry_0_t=detective_carry_0_t,
                               qa_aux=qa_aux)
        agent_carry_0_tp1 = aux['agent_carry_0_tp1']
        detective_carry_0_tp1 = aux['detective_carry_0_tp1']
        assert act.shape == (2,)
        env, obs, rew = env.step(act)
        episode["obs"] = episode["obs"].at[1 + t].set(obs)
        episode["coin_pos"] = episode["coin_pos"].at[1 + t].set(env.coin_pos)
        episode["coin_owner"] = episode["coin_owner"].at[1 + t].set(env.coin_owner)
        episode["player1_pos"] = episode["player1_pos"].at[1 + t].set(env.players_pos[0])
        episode["player2_pos"] = episode["player2_pos"].at[1 + t].set(env.players_pos[1])
        episode["act"] = episode["act"].at[t].set(act)
        episode["rew"] = episode["rew"].at[t].set(rew)
        return (env, rng, episode, t + 1, qa_auxes, agent_carry_0_tp1, detective_carry_0_tp1), aux

    (env, rng, episode, _, _, _, _), aux = jax.lax.scan(f=body_fn, init=(env, rng, episode, 0, qa_auxes, agent_carry_0_t, detective_carry_0_t), xs=(), length=trace_length)
    last_game = env
    episode['games'] = jax.tree_map(lambda x, o: x.at[trace_length].set(o), episode['games'], last_game)
    return episode, aux

def play_episode_scan_agent_and_agent_gru(env, get_actions, rng, trace_length, qa_auxes=None, agent_1_carry_0_t=None, agent_2_carry_0_t=None):
    episode = make_zero_episode(trace_length=trace_length, coin_game=env, dtype=env.dtype)

    # set initial observations
    episode["obs"] = episode["obs"].at[0].set(env.get_obs())
    episode["coin_pos"] = episode["coin_pos"].at[0].set(env.coin_pos)
    episode["coin_owner"] = episode["coin_owner"].at[0].set(env.coin_owner)
    episode["player1_pos"] = episode["player1_pos"].at[0].set(env.players_pos[0])
    episode["player2_pos"] = episode["player2_pos"].at[0].set(env.players_pos[1])

    def body_fn(carry, _):
        env, rng, episode, t, qa_auxes, agent_1_carry_0_t, agent_2_carry_0_t = carry
        qa_aux = jax.tree_map(lambda x: x[t], qa_auxes) if qa_auxes is not None else None
        rng, subrng = rax.split(rng)
        old_game = env
        episode['games'] = jax.tree_map(lambda x, o: x.at[t].set(o), episode['games'], env)
        act, aux = get_actions(subrng,
                               env,
                               episode,
                               t,
                               agent_1_carry_0_t=agent_1_carry_0_t,
                               agent_2_carry_0_t=agent_2_carry_0_t)

        agent_1_carry_0_tp1 = aux['agent_1_carry_0_tp1']
        agent_2_carry_0_tp1 = aux['agent_2_carry_0_tp1']
        assert act.shape == (2,)
        env, obs, rew = env.step(act)
        episode["obs"] = episode["obs"].at[1 + t].set(obs)
        episode["coin_pos"] = episode["coin_pos"].at[1 + t].set(env.coin_pos)
        episode["coin_owner"] = episode["coin_owner"].at[1 + t].set(env.coin_owner)
        episode["player1_pos"] = episode["player1_pos"].at[1 + t].set(env.players_pos[0])
        episode["player2_pos"] = episode["player2_pos"].at[1 + t].set(env.players_pos[1])
        episode["act"] = episode["act"].at[t].set(act)
        episode["rew"] = episode["rew"].at[t].set(rew)
        return (env, rng, episode, t + 1, qa_auxes, agent_1_carry_0_tp1, agent_2_carry_0_tp1), aux

    (env, rng, episode, _, _, _, _), aux = jax.lax.scan(f=body_fn, init=(env, rng, episode, 0, qa_auxes, agent_1_carry_0_t, agent_2_carry_0_t), xs=(), length=trace_length)
    last_game = env
    episode['games'] = jax.tree_map(lambda x, o: x.at[trace_length].set(o), episode['games'], last_game)
    return episode, aux


def get_episode_between_random_agents(env, rng):
    def get_actions(subrng, env, episode, t):
        return rax.randint(subrng, shape=(2,), minval=0, maxval=env.NUM_ACTIONS, dtype="int32"), ()

    episode, _ = play_episode_scan(env=env, get_actions=get_actions, rng=rng, trace_length=env.trace_length)
    return episode


def get_just_important_episodes(env, episodes, criterion: str):
    def is_important(episode):
        if criterion == "none":
            return True
        if criterion == "three_coin_takings":
            player1_obs = episode["obs"][:, 0]
            matrices = coin_taking_summary_matrix(player_obs=player1_obs, coin_game_template=env)
            end_matrix = matrices[-1]
            total_coin_takings = end_matrix.sum(axis=[-1, -2])
            return total_coin_takings > 3

    flags = jax.vmap(lambda e: is_important(e))(episodes)
    return jax.tree_util.tree_map(lambda x: x[flags], episodes)


def get_episodes_between_random_agents(rng, game_template: CoinGame, batch_size: int):
    rng, *rngs = rax.split(rng, batch_size + 1)
    rngs = jp.stack(rngs, axis=0)
    games = jax.vmap(lambda r: CoinGame.init(rng=r,
                                             height=game_template.HEIGHT,
                                             width=game_template.WIDTH,
                                             gnumactions=game_template.NUM_ACTIONS,
                                             trace_length=game_template.trace_length,
                                             new_coin_every_turn=game_template.new_coin_every_turn,
                                             dtype='float32')[0])(rngs)
    rngs = rax.split(rng, batch_size)
    episodes = jax.vmap(lambda g, r: get_episode_between_random_agents(g, r))(games, rngs)
    return episodes


def get_agent_obs_from_episode(rng, episode, agent_obs_function, sampling_strategy: str):
    T = episode["obs"].shape[0]  # trace_length+1
    agent_obs = agent_obs_function(episode)  # [trace_length+1, obs_dim]
    if sampling_strategy == "random":
        rng, subrng = rax.split(rng)
        idx = rax.randint(subrng, shape=(), minval=0, maxval=T, dtype="int32")
        return agent_obs[idx]


def test_get_episodes_between_random_agents():
    game_template = CoinGame(rng=jax.random.PRNGKey(0),
                             HEIGHT=2,
                             WIDTH=2,
                             NUM_ACTIONS=4,
                             coin_owner=0,
                             coin_pos=jp.array([0, 0]),
                             players_pos=jp.array([[0, 1], [1, 1]]),
                             ACTION_NAMES=tuple("left right up down".split()),
                             OBS_SHAPE=(4, 2, 2),
                             trace_length=8,
                             new_coin_every_turn=False,
                             )
    episodes = get_episodes_between_random_agents(rng=rax.PRNGKey(0), game_template=game_template, batch_size=2000)
    stats = episode_stats(episodes, game_template)
    print('random episode stats', stats)
    episodes = get_just_important_episodes(env=game_template, episodes=episodes, criterion="three_coin_takings")
    stats = episode_stats(episodes, game_template)
    print('important episode stats', stats)
    agent_obs_function = lambda e: game_template.preprocess_player_obs(episode=e, player=0, horizon=-1, config={'mode': 'minimal'})
    rng, *rngs = rax.split(rax.PRNGKey(0), episodes['obs'].shape[0] + 1)
    rngs = jp.stack(rngs, axis=0)
    agent_obs = jax.vmap(lambda e, r: get_agent_obs_from_episode(rng=r, episode=e, agent_obs_function=agent_obs_function, sampling_strategy="random"))(episodes, rngs)
    return agent_obs


def test_if_jax_performance_drops_with_passing_whole_episode():
    game_template = CoinGame(rng=jax.random.PRNGKey(0),
                             HEIGHT=2,
                             WIDTH=2,
                             NUM_ACTIONS=4,
                             coin_owner=0,
                             coin_pos=jp.array([0, 0]),
                             players_pos=jp.array([[0, 1], [1, 1]]),
                             ACTION_NAMES=tuple("left right up down".split()),
                             OBS_SHAPE=(4, 2, 2),
                             trace_length=8,
                             new_coin_every_turn=False,
                             )
    episodes = get_episodes_between_random_agents(rng=rax.PRNGKey(0), game_template=game_template, batch_size=16000)

    def ppo(e):
        return game_template.preprocess_player_obs(episode=e, player=0, horizon=-1, config={'mode': 'minimal'})

    def ppo_t(e, t):
        return game_template.preprocess_player_obs(episode=e, player=0, horizon=-1, config={'mode': 'minimal'})[t]

    def ppo_dumb(e):
        ts = jp.arange(e['obs'].shape[0])
        return jax.vmap(lambda t: ppo_t(e, t))(ts)

    # capture current time
    vmap_ppo_dumb = jax.vmap(ppo_dumb)
    t0 = time.time()
    res_ppo_dumb = vmap_ppo_dumb(episodes)
    t1 = time.time()
    print('time taken for vmap dumb', t1 - t0)
    vmap_ppo = jax.vmap(ppo)
    t0 = time.time()
    res_ppo = jax.vmap(ppo)(episodes)
    t1 = time.time()
    print('time taken for vmap', t1 - t0)
    jit_vmap_ppo_dumb = jax.jit(vmap_ppo_dumb)
    t0 = time.time()
    res_ppo_dumb = jit_vmap_ppo_dumb(episodes)
    t1 = time.time()
    print('time taken for jit vmap dumb', t1 - t0)
    jit_vmap_ppo = jax.jit(vmap_ppo)
    t0 = time.time()
    res_ppo = jit_vmap_ppo(episodes)
    t1 = time.time()
    print('time taken for jit vmap', t1 - t0)
    assert jp.allclose(res_ppo, res_ppo_dumb)


def episode_stats(episode, coin_game):
    # episode["rew"] shape [batch, time, player]
    B = episode["rew"].shape[0]
    T = episode["rew"].shape[1]
    pickups = (episode["rew"] != 0).any(axis=-1)
    # measure adversarial pickups as fraction of all pickups/coins
    adversarial_pickups = (episode["rew"] < 0).any(axis=-1)
    adversarial_frac = (pickups & adversarial_pickups).sum() / jp.maximum(1, pickups.sum())
    mean_rewards = episode["rew"].mean(axis=(0, 1))
    # also measure mean reward given pickup
    pickup_mask = pickups.astype("int32")[..., None]
    mean_pickup_rewards = ((pickup_mask * episode["rew"]).sum(axis=(0, 1))
                           / jp.maximum(1., pickup_mask.sum(axis=(0, 1))))

    # measure action variability to show when things get stuck
    one_hot_actions = jp.eye(coin_game.NUM_ACTIONS, dtype="float32")[episode["act"]]
    marginal_policies = one_hot_actions.mean(axis=(0, 1))  # [player, action]
    action_entropy = -jp.where(marginal_policies == 0, 0.,
                               marginal_policies * jp.log(marginal_policies + eps)).sum(axis=-1)

    # show simple failure to pick up own coin if adjacent
    @jax.vmap  # batch
    @jax.vmap  # time
    def compute_adjacent(prevobs, nextobs):
        prevplayers = prevobs[:coin_game.NUM_PLAYERS].astype("bool")  # [player, *space]
        nextplayers = nextobs[:coin_game.NUM_PLAYERS].astype("bool")
        prevcoins = prevobs[coin_game.NUM_PLAYERS:].astype("bool")  # [player, *space]
        nextcoins = nextobs[coin_game.NUM_PLAYERS:].astype("bool")

        neighbors = []
        for sign in [-1, +1]:
            for axis in [1, 2]:
                neighbors.append(jp.roll(prevplayers, sign, axis=axis).astype("bool"))
        reach = jp.stack(neighbors).any(axis=0)  # [player, *space]

        selfadjacent = (reach & prevcoins).any(axis=(1, 2))
        selfpickup = (nextplayers & prevcoins).any(axis=(1, 2))
        otheradjacent = (reach & prevcoins[::-1]).any(axis=(1, 2))
        otherpickup = (nextplayers & prevcoins[::-1]).any(axis=(1, 2))
        return selfadjacent, selfpickup, otheradjacent, otherpickup

    # [batch, time, player]
    selfadjacent, selfpickup, otheradjacent, otherpickup = compute_adjacent(episode["obs"][:, :-1, 0], episode["obs"][:, 1:, 0])
    # [player]
    easymisses = ((selfadjacent & ~selfpickup).astype("int32").sum(axis=(0, 1))
                  / jp.maximum(1, selfadjacent.astype("int32").sum(axis=(0, 1))))
    nearpasses = ((otheradjacent & ~otherpickup).astype("int32").sum(axis=(0, 1))
                  / jp.maximum(1, otheradjacent.astype("int32").sum(axis=(0, 1))))

    # for each player, fraction of their pickups that was adversarial
    total_timesteps = B * T
    anypickup = selfpickup | otherpickup

    total_other_pickups = (otherpickup.astype("int32").sum(axis=(0, 1)))
    total_any_pickups = (anypickup.astype("int32").sum(axis=(0, 1)))
    total_own_pickups = (selfpickup.astype("int32").sum(axis=(0, 1)))

    adversarial_pickup_div_timesteps = total_other_pickups / total_timesteps
    any_pickup_div_timesteps = total_any_pickups / total_timesteps
    adversarial_pickup_div_all_pickup = total_other_pickups / jp.maximum(total_any_pickups, 1)
    own_pickup_div_timesteps = total_own_pickups / total_timesteps

    # old logs, TODO : remove after checking it is equal to adversarial_pickup_div_timesteps
    adversity = (otherpickup.astype("int32").sum(axis=(0, 1))
                 / jp.maximum(1, anypickup.astype("int32")).sum(axis=(0, 1)))

    stats = dict(mean_rewards=mean_rewards,
                 mean_pickup_rewards=mean_pickup_rewards,
                 action_entropy=action_entropy,
                 easymisses=easymisses,
                 adversity=adversity,
                 adversarial_pickup_div_timesteps=adversarial_pickup_div_timesteps,
                 any_pickup_div_timesteps=any_pickup_div_timesteps,
                 adversarial_pickup_div_all_pickup=adversarial_pickup_div_all_pickup,
                 own_pickup_div_timesteps=own_pickup_div_timesteps,
                 nearpasses=nearpasses,
                 )
    return stats


def mirror_episode(episode):
    mirrored_coin_pos = episode['coin_pos']  # important the two axes should not be flipped because they are just x and y indices
    mirrored_coin_owner = 1 - episode['coin_owner']
    mirrored_act = jp.flip(episode['act'], axis=-1)
    mirrored_rew = jp.flip(episode['rew'], axis=-1)
    mirrored_obs = jp.flip(episode['obs'], axis=1)  # (t, player, *obs_shape)
    mirrored_player1_pos = episode['player2_pos']
    mirrored_player2_pos = episode['player1_pos']
    # assert set(episode.keys()) == {'coin_pos', 'coin_owner', 'act', 'rew', 'obs', 'player1_pos', 'player2_pos'}
    mirrored_episode = {'coin_pos': mirrored_coin_pos,
                        'coin_owner': mirrored_coin_owner,
                        'act': mirrored_act,
                        'rew': mirrored_rew,
                        'obs': mirrored_obs,
                        'player1_pos': mirrored_player1_pos,
                        'player2_pos': mirrored_player2_pos}
    return mirrored_episode


def test_painting_episodes():
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
    episodes = get_episodes_between_random_agents(rng=rax.PRNGKey(0), game_template=game_template, batch_size=2000)
    print(game_template.printepisode(episodes,
                                     batch_max=2,
                                     time_max=16,
                                     ))


def test_juan_state():
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
                             prev_coin_takers=jp.array([-1, -1]),
                             )
    episodes = get_episodes_between_random_agents(rng=rax.PRNGKey(0), game_template=game_template, batch_size=1)
    juan = jax.vmap(lambda e: game_template.preprocess_player_obs(episode=e,
                                                                  player=0,
                                                                  horizon=-1,
                                                                  config={'mode': 'juan'}, ))(episodes)


    # compare with old juan bits
    juan_old = jax.vmap(lambda e: game_template.preprocess_player_obs(episode=e,
                                                                        player=0,
                                                                        horizon=-1,
                                                                        config={'mode': 'old_juan'}, ))(episodes)

    assert jp.allclose(juan, juan_old)

    juan = jax.vmap(lambda e: game_template.preprocess_player_obs(episode=e,
                                                                  player=1,
                                                                  horizon=-1,
                                                                  config={'mode': 'juan'}, ))(episodes)

    # compare with old juan bits
    juan_old = jax.vmap(lambda e: game_template.preprocess_player_obs(episode=e,
                                                                      player=1,
                                                                      horizon=-1,
                                                                      config={'mode': 'old_juan'}, ))(episodes)

    assert jp.allclose(juan, juan_old)


    print('done')


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import io

def buffer_plot_and_get(fig):
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def plot_game(game):
    width = game['games'].WIDTH
    height = game['games'].HEIGHT
    coin_position = game['coin_pos'][0], game['coin_pos'][1]
    coin_owner = game['coin_owner']
    player1_pos = game['player1_pos'][0], game['player1_pos'][1]
    player2_pos = game['player2_pos'][0], game['player2_pos'][1]
    # set the position of the coin
    coin_x = coin_position[1]
    coin_y = coin_position[0]
    player1_y = player1_pos[0]
    player1_x = player1_pos[1]
    player2_y = player2_pos[0]
    player2_x= player2_pos[1]

    # create a figure and axis object
    fig, ax = plt.subplots()
    # loop over the board and add the squares to the plot
    for i in range(height):
        for j in range(width):
            ax.add_patch(plt.Rectangle((i,j), 1, 1, fill=False, color='black', linewidth=3.))

    # add the coin to the plot
    if coin_owner == 0:
        ax.add_patch(plt.Circle((coin_x+0.8,coin_y+0.8), 0.2, fill=True, color='red'))
    elif coin_owner == 1:
        ax.add_patch(plt.Circle((coin_x+0.8,coin_y+0.8), 0.2, fill=True, color='blue'))

    # load the player images
    player1_img = mpimg.imread('/Users/miladaghajohari/PycharmProjects/extramodels/assets/player_1.png')
    player2_img = mpimg.imread('/Users/miladaghajohari/PycharmProjects/extramodels/assets/player_2.png')
    # red player
    ax.imshow(player1_img, extent=[player1_x, player1_x+0.4, player1_y, player1_y+0.4], alpha=1.0)
    # blue player
    ax.imshow(player2_img, extent=[player2_x+0.6, player2_x+1.0, player2_y, player2_y+0.4], alpha=1.0)

    # set the limits of the plot
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # hide the ticks and axis labels
    ax.set_xticks([])
    ax.set_yticks([])

    return fig
#%%
def episodes_to_grid_image(episodes, max_batch):
    batch_size = episodes['act'].shape[0]
    if batch_size > max_batch:
        batch_size = max_batch
        episodes = jax.tree_map(lambda x: x[:batch_size], episodes)

    game_sample = jax.tree_map(lambda x: x[0], episodes)
    trace_length = game_sample['coin_pos'].shape[0]
    def complete_episodes_to_images(games):
        images = [buffer_plot_and_get(plot_game(jax.tree_map(lambda x: x[i], games))) for i in range(trace_length)]
        return images
    images = [complete_episodes_to_images(jax.tree_map(lambda x: x[i], episodes)) for i in range(batch_size)]

    num_rows = len(images)
    num_columns = len(images[0])

    # Get the size of one image
    image_width, image_height = images[0][0].size

    # Create a new image that will hold the grid
    grid_width = image_width * num_columns
    grid_height = image_height * num_rows
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Iterate over the images and paste them into the grid image
    for row in range(num_rows):
        for col in range(num_columns):
            image = images[row][col]
            x_offset = col * image_width
            y_offset = row * image_height
            grid_image.paste(image, (x_offset, y_offset))

    return grid_image


if __name__ == '__main__':
    # test_distance_function()
    # test_play_episode()
    # random_obs = test_get_episodes_between_random_agents()
    # test_if_jax_performance_drops_with_passing_whole_episode()
    # test_smooth_obs()
    # test_painting_episodes()
    test_juan_state()
