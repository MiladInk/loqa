import dataclasses
import time
from functools import partial
from typing import Tuple
import tabulate
import jax
import numpy as np
from flax import struct as struct
from jax import random as rax, numpy as jp

from utils import rscope
import wandb

eps = 1e-8
MOVES = jp.array([[0, -1], [0, +1], [-1, 0], [+1, 0]])

# NOTE difference from LOLA paper coin game: players/coins may spawn on
# top of each other

# taken from https://github.com/google/flax/issues/3032
def dataclass_eq_with_arrays(cls):
  def eq(a, b):
    if type(a) is not type(b):
      return False
    for f in dataclasses.fields(a):
      fa = getattr(a, f.name)
      fb = getattr(b, f.name)
      if isinstance(fa, (np.ndarray, jp.ndarray)):
        if not (fa == fb).all():
          return False
      else:
        if fa != fb:
          return False
    return True
  cls = struct.dataclass(cls)
  cls.__eq__ = eq
  return cls

@dataclass_eq_with_arrays
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

def make_zero_episode(trace_length, coin_game, dtype='float32'):
    # get something with the same structure as play_episode would return
    obs = jp.zeros([1 + trace_length, 2, *coin_game.OBS_SHAPE], dtype=dtype)
    act = jp.zeros([trace_length, 2], dtype="int32")
    logp = jp.zeros([trace_length, 2, coin_game.NUM_ACTIONS], dtype=dtype)
    rew = jp.zeros([trace_length, 2], dtype=dtype)
    coin_pos = jp.zeros([trace_length + 1, 2], dtype="int32")
    coin_owner = jp.zeros([trace_length + 1, 1], dtype="int32")
    player1_pos = jp.zeros([trace_length + 1, 2], dtype="int32")
    player2_pos = jp.zeros([trace_length + 1, 2], dtype="int32")
    games = jax.tree_map(lambda x: jp.expand_dims(x, axis=0).repeat(trace_length + 1, axis=0), coin_game)
    return dict(obs=obs,
                act=act,
                rew=rew,
                coin_pos=coin_pos,
                coin_owner=coin_owner,
                player1_pos=player1_pos,
                player2_pos=player2_pos,
                games=games,
                logp=logp, )

def coin_game_params(hp):
    return dict(
        height=hp['game']['height'],
        width=hp['game']['width'],
        gnumactions=hp['game']['gnumactions'],
        trace_length=hp['game']['game_length'],
        new_coin_every_turn=False,
        dtype='float32',
    )

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

def get_new_distances(episode, t, hp, rng, distance_of_this_player):
    other_player_pos = jax.lax.select(distance_of_this_player == 1, episode['player2_pos'][t], episode['player1_pos'][t])
    coin_pos = episode['coin_pos'][t]

    def get_new_distance(a):
        move = MOVES[a]
        new_agent_pos_x = (other_player_pos[0] + move[0]) % hp['game']['height']
        new_agent_pos_y = (other_player_pos[1] + move[1]) % hp['game']['width']
        new_dif_x = jp.abs(new_agent_pos_x - coin_pos[0])
        new_dif_y = jp.abs(new_agent_pos_y - coin_pos[1])
        wrapped_dif_x = jp.minimum(new_dif_x, hp['game']['height'] - new_dif_x)
        wrapped_dif_y = jp.minimum(new_dif_y, hp['game']['width'] - new_dif_y)
        return {'action': a, 'new_distance': wrapped_dif_x + wrapped_dif_y}

    new_distances = jax.vmap(lambda a: get_new_distance(a))(jp.arange(hp['game']['gnumactions']))
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

@partial(jax.jit, static_argnames=('hp',))
def do_eval_agent_against_always_cooperate(agent, hp, rng):
    assert agent.player == 0, 'only works for when agent is player 0'
    assert hp['game']['gnumactions'] == 4, 'only works for 4 actions case'
    other_player = 1 - agent.player

    def get_actions(subrng, env, episode, t):
        subrngs = jax.random.split(subrng, 2)
        agent_action = agent.get_action(subrngs[0], episode, t)
        cooperative_action = get_cooperative_action(episode=episode, t=t, hp=hp, rng=rscope(rng, 'cooperative_action_rng'), agent_player=agent.player, other_player=other_player)
        actions = jp.stack([agent_action, cooperative_action])
        return actions, ()

    init_rng = rscope(rng, 'game_init')
    env, _ = CoinGame.init(rng=init_rng, **coin_game_params(hp))
    game_play_rng = rscope(rng, 'game_play')
    episode, aux = play_episode_scan(env=env,
                                     get_actions=get_actions,
                                     rng=game_play_rng,
                                     trace_length=hp['game']['game_length'],
                                     qa_auxes=None,
                                     )

    return episode


def eval_agent_against_always_cooperate(state, hp, jit_episode_stats, player):
    agent = state[f'agent{player}']
    assert agent.player == 0, 'only works for when agent is player 0'
    assert hp['game']['gnumactions'] == 4, 'only works for 4 actions case'
    agent_player = agent.player
    other_player = 1 - agent.player

    play_rngs = jax.random.split(state['rng'], hp['batch_size'])
    episodes = jax.vmap(lambda r: do_eval_agent_against_always_cooperate(agent=agent,
                                                                         hp=hp,
                                                                         rng=r))(play_rngs)

    stats = jit_episode_stats(episodes)

    wandb.log({f'always_cooperate_{player}_mean_rewards_a': stats["mean_rewards"][agent_player],
               f'always_cooperate_{player}_mean_rewards_d': stats["mean_rewards"][other_player],
               f'always_cooperate_{player}_adversity_a': stats["adversity"][agent_player],
               f'always_cooperate_{player}_adversity_d': stats["adversity"][other_player],
               f'always_cooperate_{player}_easymisses_a': stats["easymisses"][agent_player],
               f'always_cooperate_{player}_easymisses_d': stats["easymisses"][other_player],
               f'always_cooperate_{player}_adversarial_pickup_div_timesteps_a': stats["adversarial_pickup_div_timesteps"][agent_player],
               f'always_cooperate_{player}_adversarial_pickup_div_timesteps_d': stats["adversarial_pickup_div_timesteps"][other_player],
               f'always_cooperate_{player}_any_pickup_div_timesteps_a': stats["any_pickup_div_timesteps"][agent_player],
               f'always_cooperate_{player}_any_pickup_div_timesteps_d': stats["any_pickup_div_timesteps"][other_player],
               f'always_cooperate_{player}_adversarial_pickup_div_all_pickup_a': stats["adversarial_pickup_div_all_pickup"][agent_player],
               f'always_cooperate_{player}_adversarial_pickup_div_all_pickup_d': stats["adversarial_pickup_div_all_pickup"][other_player],
               f'always_cooperate_{player}_nearpasses_a': stats["nearpasses"][agent_player],
               f'always_cooperate_{player}_nearpasses_d': stats["nearpasses"][other_player],
               f'always_cooperate_{player}_own_pickup_div_timesteps_a': stats["own_pickup_div_timesteps"][agent_player],
               f'always_cooperate_{player}_own_pickup_div_timesteps_d': stats["own_pickup_div_timesteps"][other_player],
               },
              step=state['step'])

    return episodes

@partial(jax.jit, static_argnames=('hp',))
def do_eval_agent_against_always_defect(agent, hp, rng):
    assert agent.player == 0, 'only works for when agent is player 0'
    assert hp['game']['gnumactions'] == 4, 'only works for 4 actions case'
    other_player = 1 - agent.player

    def get_actions(subrng, env, episode, t):
        subrngs = jax.random.split(subrng, 2)
        agent_action = agent.get_action(subrngs[0], episode, t)
        defect_action = get_defect_action(episode=episode, t=t, hp=hp, rng=rscope(rng, 'defect_action_rng'), agent_player=agent.player, other_player=other_player)
        actions = jp.stack([agent_action, defect_action])
        return actions, ()

    init_rng = rscope(rng, 'game_init')
    env, _ = CoinGame.init(rng=init_rng, **coin_game_params(hp))
    game_play_rng = rscope(rng, 'game_play')
    episode, aux = play_episode_scan(env=env,
                                     get_actions=get_actions,
                                     rng=game_play_rng,
                                     trace_length=hp['game']['game_length'],
                                     qa_auxes=None,
                                     )

    return episode

def eval_agent_against_always_defect(state, hp, jit_episode_stats, player):
    agent = state[f'agent{player}']
    assert agent.player == 0, 'only works for when agent is player 0'
    assert hp['game']['gnumactions'] == 4, 'only works for 4 actions case'
    agent_player = agent.player
    other_player = 1 - agent.player

    play_rngs = jax.random.split(state['rng'], hp['batch_size'])
    episodes = jax.vmap(lambda r: do_eval_agent_against_always_defect(agent=agent,
                                                                      hp=hp,
                                                                      rng=r))(play_rngs)

    stats = jit_episode_stats(episodes)

    wandb.log({f'always_defect_{player}_mean_rewards_a': stats["mean_rewards"][agent_player],
               f'always_defect_{player}_mean_rewards_d': stats["mean_rewards"][other_player],
               f'always_defect_{player}_adversity_a': stats["adversity"][agent_player],
               f'always_defect_{player}_adversity_d': stats["adversity"][other_player],
               f'always_defect_{player}_easymisses_a': stats["easymisses"][agent_player],
               f'always_defect_{player}_easymisses_d': stats["easymisses"][other_player],
               f'always_defect_{player}_adversarial_pickup_div_timesteps_a': stats["adversarial_pickup_div_timesteps"][agent_player],
               f'always_defect_{player}_adversarial_pickup_div_timesteps_d': stats["adversarial_pickup_div_timesteps"][other_player],
               f'always_defect_{player}_any_pickup_div_timesteps_a': stats["any_pickup_div_timesteps"][agent_player],
               f'always_defect_{player}_any_pickup_div_timesteps_d': stats["any_pickup_div_timesteps"][other_player],
               f'always_defect_{player}_adversarial_pickup_div_all_pickup_a': stats["adversarial_pickup_div_all_pickup"][agent_player],
               f'always_defect_{player}_adversarial_pickup_div_all_pickup_d': stats["adversarial_pickup_div_all_pickup"][other_player],
               f'always_defect_{player}_nearpasses_a': stats["nearpasses"][agent_player],
               f'always_defect_{player}_nearpasses_d': stats["nearpasses"][other_player],
               f'always_defect_{player}_own_pickup_div_timesteps_a': stats["own_pickup_div_timesteps"][agent_player],
               f'always_defect_{player}_own_pickup_div_timesteps_d': stats["own_pickup_div_timesteps"][other_player],
               },
              step=state['step'])

    return episodes

