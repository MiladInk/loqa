import dataclasses
import time
from functools import partial
from typing import Tuple
import tabulate
import jax
import numpy as np
from flax import struct as struct
from jax import random as rax, numpy as jp

from coin_agent import CoinAgent, GRUCoinAgent
from utils import rscope
import wandb
import flax

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
class ThreePlayerCoinGame:
    rng: "PRNGKey"
    coin_owner: "int[]"
    coin_pos: "int[2]"
    players_pos: "int[NUM_PLAYERS]"
    trace_length: int = struct.field(pytree_node=False)

    HEIGHT: int = struct.field(pytree_node=False)
    WIDTH: int = struct.field(pytree_node=False)
    ACTION_NAMES = tuple("left right up down".split())
    NUM_ACTIONS = 4
    OBS_SHAPE: Tuple[int, int, int] = struct.field(pytree_node=False)
    NUM_PLAYERS = 3
    SYMBOLS = tuple("XOYxoy")

    @classmethod
    def init(cls, rng, height, width, trace_length):
        rng, owner_rng, position_rng = rax.split(rng, 3)
        coin_owner = rax.randint(owner_rng, minval=0, maxval=2, shape=[])
        players_pos = jax.vmap(lambda r: cls._random_position(r, height, width))(rax.split(position_rng, cls.NUM_PLAYERS))
        coin_pos = cls._random_coin_position(rscope(position_rng, "coin"), players_pos, height, width)
        assert height * width >= 2, "board too small for the coin not to land on players"

        env = cls(rng=rng,
                  coin_owner=coin_owner,
                  coin_pos=coin_pos,
                  players_pos=players_pos,
                  OBS_SHAPE=(2 * cls.NUM_PLAYERS, height, width),
                  HEIGHT=height,
                  WIDTH=width,
                  trace_length=trace_length,
                  )

        obs = env.get_obs()
        return env, obs

    def step(self, actions):
        # takers get 1, coin_owner gets -2 if any other taker
        assert actions.shape == (self.NUM_PLAYERS,)
        moves = MOVES[actions]
        new_players_pos = (self.players_pos + moves) % jp.array([self.HEIGHT, self.WIDTH])[None]
        takers = (new_players_pos == self.coin_pos[None]).all(axis=1)

        assert new_players_pos.shape == self.players_pos.shape
        owner = jp.eye(self.NUM_PLAYERS, dtype="bool")[self.coin_owner]
        rewards = 1 * takers - 2 * owner * (takers & ~owner).any()

        new_rng, rng = rax.split(self.rng)
        new_coin_pos = self._random_coin_position(new_rng, new_players_pos, self.HEIGHT, self.WIDTH)
        new_coin_owner = (self.coin_owner + 1) % self.NUM_PLAYERS
        new_coin_pos = jp.where(takers.any(), new_coin_pos, self.coin_pos)
        new_coin_owner = jp.where(takers.any(), new_coin_owner, self.coin_owner)

        new_rng, rng = rax.split(self.rng)

        env = self.replace(
            rng=new_rng,
            players_pos=new_players_pos,
            coin_pos=new_coin_pos,
            coin_owner=new_coin_owner,
        )
        obs = env.get_obs()
        return env, obs, rewards

    @classmethod
    def _random_position(cls, rng, height, width):
        pos = rax.randint(rng, minval=0, maxval=height * width, shape=[])
        return jp.array([pos // width, pos % width])

    @classmethod
    def _random_coin_position(cls, rng, players_pos, height, width):
        # like LOLA/POLA, sample coin so that it is never in the same place as any player
        assert height * width > cls.NUM_PLAYERS  # only possible if there is room
        players_pos_flat = players_pos[:, 0] * width + players_pos[:, 1]
        occupied = (players_pos_flat[:, None] == jp.arange(height * width)[None, :]).any(axis=0)
        coin_pos_flat = rax.choice(rng, occupied.size, p=1 - occupied, shape=[])
        coin_pos = jp.stack([coin_pos_flat // width, coin_pos_flat % width])
        return coin_pos

    def get_obs(self):
        x = jp.zeros(self.OBS_SHAPE)
        x = x.at[jp.arange(self.NUM_PLAYERS), self.players_pos[:, 0], self.players_pos[:, 1]].set(1.)
        x = x.at[self.NUM_PLAYERS + self.coin_owner, self.coin_pos[0], self.coin_pos[1]].set(1.)
        # x's structure is [0:player_0_pos, 1:player_1_pos, 2:player_2_pos, 3:coin_0_pos, 4:coin_1_pos, 5:coin_2_pos]
        # make sure for each player, channel 0 corresponds to the player's own position and channel 3 to their coin
        x_player_0 = x
        x_player_1 = x[((1, 2, 0, 4, 5, 3),)]  # [player_1_pos, player_2_pos, player_0_pos, coin_1_pos, coin_2_pos, coin_0_pos]
        x_player_2 = x[((2, 0, 1, 5, 3, 4),)]  # [player_2_pos, player_0_pos, player_1_pos, coin_2_pos, coin_0_pos, coin_1_pos]
        return jp.stack([x_player_0, x_player_1, x_player_2])

    def renderframe(self, obs, act, rew):

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


def make_zero_episode(trace_length, coin_game):
    # get something with the same structure as play_episode would return
    obs = jp.zeros([1 + trace_length, coin_game.NUM_PLAYERS, *coin_game.OBS_SHAPE])
    act = jp.zeros([trace_length, 3], dtype="int32")
    logp = jp.zeros([trace_length, 3, coin_game.NUM_ACTIONS])
    rew = jp.zeros([trace_length, 3])
    coin_pos = jp.zeros([trace_length + 1, 2], dtype="int32")
    coin_owner = jp.zeros([trace_length + 1, 1], dtype="int32")
    player1_pos = jp.zeros([trace_length + 1, 2], dtype="int32")
    player2_pos = jp.zeros([trace_length + 1, 2], dtype="int32")
    player3_pos = jp.zeros([trace_length + 1, 2], dtype="int32")
    games = jax.tree_map(lambda x: jp.expand_dims(x, axis=0).repeat(trace_length + 1, axis=0), coin_game)
    return dict(obs=obs,
                act=act,
                rew=rew,
                coin_pos=coin_pos,
                coin_owner=coin_owner,
                player1_pos=player1_pos,
                player2_pos=player2_pos,
                player3_pos=player3_pos,
                games=games,
                logp=logp, )


def coin_game_params(hp):
    return dict(
        height=hp['game']['height'],
        width=hp['game']['width'],
        trace_length=hp['game']['game_length'],
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
    player_pos = jax.lax.select_n(distance_of_this_player,
                                      episode['player1_pos'][t],
                                      episode['player2_pos'][t],
                                      episode['player3_pos'][t])
    coin_pos = episode['coin_pos'][t]

    def get_new_distance(a):
        move = MOVES[a]
        new_agent_pos_x = (player_pos[0] + move[0]) % hp['game']['height']
        new_agent_pos_y = (player_pos[1] + move[1]) % hp['game']['width']
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


def get_cooperative_action(episode, t, hp, rng, the_cooperative_player):
    sorted_distances = get_new_distances(episode, t, hp, rng, distance_of_this_player=the_cooperative_player)['sorted_distances']
    coin_owner = episode['coin_owner'][t]
    cooperative_action = jp.where(coin_owner == the_cooperative_player, sorted_distances['action'][0], sorted_distances['action'][-1]).reshape()

    return cooperative_action


def get_defect_action(episode, t, hp, rng, the_defective_player):
    sorted_distances = get_new_distances(episode, t, hp, rng, distance_of_this_player=the_defective_player)['sorted_distances']
    defect_action = sorted_distances['action'][0].reshape()

    return defect_action


def play_episode_scan(env, get_actions, rng, trace_length):
    episode = make_zero_episode(trace_length=trace_length, coin_game=env)

    # set initial observations
    episode["obs"] = episode["obs"].at[0].set(env.get_obs())
    episode["coin_pos"] = episode["coin_pos"].at[0].set(env.coin_pos)
    episode["coin_owner"] = episode["coin_owner"].at[0].set(env.coin_owner)
    episode["player1_pos"] = episode["player1_pos"].at[0].set(env.players_pos[0])
    episode["player2_pos"] = episode["player2_pos"].at[0].set(env.players_pos[1])
    episode["player3_pos"] = episode["player3_pos"].at[0].set(env.players_pos[2])

    def body_fn(carry, _):
        env, rng, episode, t = carry
        rng, subrng = rax.split(rng)
        old_game = env
        episode['games'] = jax.tree_map(lambda x, o: x.at[t].set(o), episode['games'], env)
        act, aux = get_actions(subrng, env, episode, t)
        assert act.shape == (3,)
        env, obs, rew = env.step(act)
        episode["obs"] = episode["obs"].at[1 + t].set(obs)
        episode["coin_pos"] = episode["coin_pos"].at[1 + t].set(env.coin_pos)
        episode["coin_owner"] = episode["coin_owner"].at[1 + t].set(env.coin_owner)
        episode["player1_pos"] = episode["player1_pos"].at[1 + t].set(env.players_pos[0])
        episode["player2_pos"] = episode["player2_pos"].at[1 + t].set(env.players_pos[1])
        episode["player3_pos"] = episode["player3_pos"].at[1 + t].set(env.players_pos[2])
        episode["act"] = episode["act"].at[t].set(act)
        episode["rew"] = episode["rew"].at[t].set(rew)
        return (env, rng, episode, t + 1), aux

    (env, rng, episode, _,), aux = jax.lax.scan(f=body_fn, init=(env, rng, episode, 0), xs=(), length=trace_length)
    last_game = env
    episode['games'] = jax.tree_map(lambda x, o: x.at[trace_length].set(o), episode['games'], last_game)
    return episode, aux


@partial(jax.jit, static_argnames=('hp',))
def do_eval_agent_against_always_cooperate(agent, hp, rng):
    assert agent.player == 0, 'only works for when agent is player 0'
    assert hp['game']['gnumactions'] == 4, 'only works for 4 actions case'

    def get_actions(subrng, env, episode, t):
        subrngs = jax.random.split(subrng, 2)
        agent_action = agent.get_action(subrngs[0], episode, t)
        cooperative_action_agent_1 = get_cooperative_action(episode=episode, t=t, hp=hp, rng=rscope(rng, 'cooperative_action_rng_agent_1'), the_cooperative_player=1)
        cooperative_action_agent_2 = get_cooperative_action(episode=episode, t=t, hp=hp, rng=rscope(rng, 'cooperative_action_rng_agent_2'),  the_cooperative_player=2)
        actions = jp.stack([agent_action, cooperative_action_agent_1, cooperative_action_agent_2])
        return actions, ()

    init_rng = rscope(rng, 'game_init')
    env, _ = ThreePlayerCoinGame.init(rng=init_rng, **coin_game_params(hp))
    game_play_rng = rscope(rng, 'game_play')
    episode, aux = play_episode_scan(env=env,
                                     get_actions=get_actions,
                                     rng=game_play_rng,
                                     trace_length=hp['game']['game_length'],
                                     )

    return episode


def eval_agent_against_always_cooperate(state, hp, jit_episode_stats, player):
    agent = state[f'agent{player}']
    assert agent.player == 0, 'only works for when agent is player 0'
    assert hp['game']['gnumactions'] == 4, 'only works for 4 actions case'
    agent_player = agent.player

    play_rngs = jax.random.split(state['rng'], hp['batch_size'])
    episodes = jax.vmap(lambda r: do_eval_agent_against_always_cooperate(agent=agent,
                                                                         hp=hp,
                                                                         rng=r))(play_rngs)

    stats = jit_episode_stats(episodes)
    # print(stats)

    wandb.log({f'always_cooperate_{player}_mean_rewards_a0': stats["mean_rewards"][agent_player],
               f'always_cooperate_{player}_mean_rewards_d1': stats["mean_rewards"][1],
               f'always_cooperate_{player}_mean_rewards_d2': stats["mean_rewards"][2],
               f'always_cooperate_{player}_adversity_a0': stats["adversity"][agent_player],
               f'always_cooperate_{player}_adversity_d1': stats["adversity"][1],
               f'always_cooperate_{player}_adversity_d2': stats["adversity"][2],
               f'always_cooperate_{player}_easymisses_a0': stats["easymisses"][agent_player],
               f'always_cooperate_{player}_easymisses_d1': stats["easymisses"][1],
               f'always_cooperate_{player}_easymisses_d2': stats["easymisses"][2],
               f'always_cooperate_{player}_adversarial_pickup_div_timesteps_a0': stats["adversarial_pickup_div_timesteps"][agent_player],
               f'always_cooperate_{player}_adversarial_pickup_div_timesteps_d1': stats["adversarial_pickup_div_timesteps"][1],
               f'always_cooperate_{player}_adversarial_pickup_div_timesteps_d2': stats["adversarial_pickup_div_timesteps"][2],
               f'always_cooperate_{player}_any_pickup_div_timesteps_a0': stats["any_pickup_div_timesteps"][agent_player],
               f'always_cooperate_{player}_any_pickup_div_timesteps_d1': stats["any_pickup_div_timesteps"][1],
               f'always_cooperate_{player}_any_pickup_div_timesteps_d2': stats["any_pickup_div_timesteps"][2],
               f'always_cooperate_{player}_adversarial_pickup_div_all_pickup_a0': stats["adversarial_pickup_div_all_pickup"][agent_player],
               f'always_cooperate_{player}_adversarial_pickup_div_all_pickup_d1': stats["adversarial_pickup_div_all_pickup"][1],
               f'always_cooperate_{player}_adversarial_pickup_div_all_pickup_d2': stats["adversarial_pickup_div_all_pickup"][2],
               f'always_cooperate_{player}_nearpasses_a0': stats["nearpasses"][agent_player],
               f'always_cooperate_{player}_nearpasses_d1': stats["nearpasses"][1],
               f'always_cooperate_{player}_nearpasses_d2': stats["nearpasses"][2],
               f'always_cooperate_{player}_own_pickup_div_timesteps_a0': stats["own_pickup_div_timesteps"][agent_player],
               f'always_cooperate_{player}_own_pickup_div_timesteps_d1': stats["own_pickup_div_timesteps"][1],
               f'always_cooperate_{player}_own_pickup_div_timesteps_d2': stats["own_pickup_div_timesteps"][2],
               },
              step=state['step'])

    return episodes

@partial(jax.jit, static_argnames=('hp',))
def do_eval_agent_against_always_defect(agent, hp, rng):
    assert agent.player == 0, 'only works for when agent is player 0'
    assert hp['game']['gnumactions'] == 4, 'only works for 4 actions case'

    def get_actions(subrng, env, episode, t):
        subrngs = jax.random.split(subrng, 2)
        agent_action = agent.get_action(subrngs[0], episode, t)
        defect_action_agent_1 = get_defect_action(episode=episode, t=t, hp=hp, rng=rscope(rng, 'defect_action_rng_agent_1'), the_defective_player=1)
        defect_action_agent_2 = get_defect_action(episode=episode, t=t, hp=hp, rng=rscope(rng, 'defect_action_rng_agent_2'), the_defective_player=2)
        actions = jp.stack([agent_action, defect_action_agent_1, defect_action_agent_2])
        return actions, ()

    init_rng = rscope(rng, 'game_init')
    env, _ = ThreePlayerCoinGame.init(rng=init_rng, **coin_game_params(hp))
    game_play_rng = rscope(rng, 'game_play')
    episode, aux = play_episode_scan(env=env,
                                     get_actions=get_actions,
                                     rng=game_play_rng,
                                     trace_length=hp['game']['game_length'],
                                     )

    return episode


def eval_agent_against_always_defect(state, hp, jit_episode_stats, player):
    agent = state[f'agent{player}']
    assert agent.player == 0, 'only works for when agent is player 0'
    assert hp['game']['gnumactions'] == 4, 'only works for 4 actions case'
    agent_player = agent.player

    play_rngs = jax.random.split(state['rng'], hp['batch_size'])
    episodes = jax.vmap(lambda r: do_eval_agent_against_always_defect(agent=agent,
                                                                      hp=hp,
                                                                      rng=r))(play_rngs)

    stats = jit_episode_stats(episodes)
    # print(stats)

    wandb.log({f'always_defect_{player}_mean_rewards_a0': stats["mean_rewards"][agent_player],
               f'always_defect_{player}_mean_rewards_d1': stats["mean_rewards"][1],
               f'always_defect_{player}_mean_rewards_d2': stats["mean_rewards"][2],
               f'always_defect_{player}_adversity_a0': stats["adversity"][agent_player],
               f'always_defect_{player}_adversity_d1': stats["adversity"][1],
               f'always_defect_{player}_adversity_d2': stats["adversity"][2],
               f'always_defect_{player}_easymisses_a0': stats["easymisses"][agent_player],
               f'always_defect_{player}_easymisses_d1': stats["easymisses"][1],
               f'always_defect_{player}_easymisses_d2': stats["easymisses"][2],
               f'always_defect_{player}_adversarial_pickup_div_timesteps_a0': stats["adversarial_pickup_div_timesteps"][agent_player],
               f'always_defect_{player}_adversarial_pickup_div_timesteps_d1': stats["adversarial_pickup_div_timesteps"][1],
               f'always_defect_{player}_adversarial_pickup_div_timesteps_d2': stats["adversarial_pickup_div_timesteps"][2],
               f'always_defect_{player}_any_pickup_div_timesteps_a0': stats["any_pickup_div_timesteps"][agent_player],
               f'always_defect_{player}_any_pickup_div_timesteps_d1': stats["any_pickup_div_timesteps"][1],
               f'always_defect_{player}_any_pickup_div_timesteps_d2': stats["any_pickup_div_timesteps"][2],
               f'always_defect_{player}_adversarial_pickup_div_all_pickup_a0': stats["adversarial_pickup_div_all_pickup"][agent_player],
               f'always_defect_{player}_adversarial_pickup_div_all_pickup_d1': stats["adversarial_pickup_div_all_pickup"][1],
               f'always_defect_{player}_adversarial_pickup_div_all_pickup_d2': stats["adversarial_pickup_div_all_pickup"][2],
               f'always_defect_{player}_nearpasses_a0': stats["nearpasses"][agent_player],
               f'always_defect_{player}_nearpasses_d1': stats["nearpasses"][1],
               f'always_defect_{player}_nearpasses_d2': stats["nearpasses"][2],
               f'always_defect_{player}_own_pickup_div_timesteps_a0': stats["own_pickup_div_timesteps"][agent_player],
               f'always_defect_{player}_own_pickup_div_timesteps_d1': stats["own_pickup_div_timesteps"][1],
               f'always_defect_{player}_own_pickup_div_timesteps_d2': stats["own_pickup_div_timesteps"][2],
               },
              step=state['step'])

    return episodes


if __name__ == '__main__':
    wandb.init()
    env, init_obs = ThreePlayerCoinGame.init(
        rng=jax.random.PRNGKey(0),
        height=3,
        width=3,
        trace_length=50,
    )
    print(f'env = {env}')

    env, obs, rew = env.step(jp.array([0, 1, 2]))
    make_zero_episode(50, env)
    def get_actions(subrng, env, episode, t):
        return jax.random.randint(subrng, (3,), minval=0, maxval=4), ()
    episode, aux = play_episode_scan(env, get_actions, jax.random.PRNGKey(0), 50)
    print(f'episode = {episode}')

    dummy_rng = rax.PRNGKey(0)
    rng1 = rax.PRNGKey(1)
    dummy_env = env
    dummy_episode = make_zero_episode(trace_length=50, coin_game=dummy_env)
    dummy_obs_seq = dummy_episode['obs'][:, 0].reshape(dummy_episode['obs'].shape[0], -1)
    agent_module = GRUCoinAgent(hidden_size_actor=32,
                         hidden_size_qvalue=32,
                         layers_before_gru_actor=2,
                         layers_before_gru_qvalue=2,)
    agent_params = agent_module.init(rng1, {'obs_seq': dummy_obs_seq, 'rng': dummy_rng, 't': 0})
    agent = CoinAgent(params=agent_params, model=agent_module, player=0)
    state = {'agent0':  agent, 'rng': dummy_rng, 'step': 0}
    hp = {'game': {'height': 3, 'width': 3, 'gnumactions': 4, 'game_length': 50}, 'batch_size': 7}
    hp = flax.core.FrozenDict(hp)
    episode_stats_jitted = jax.jit(lambda es: episode_stats(es, dummy_env))
    eval_agent_against_always_defect(state, hp, episode_stats_jitted, player=0)
    eval_agent_against_always_cooperate(state, hp, episode_stats_jitted, player=0)
