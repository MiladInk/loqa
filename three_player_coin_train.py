import collections
import os
import pickle
import random
import shutil
import time
from datetime import datetime
from functools import partial
from typing import Any

import flax as flax
import hydra
import jax
import jax.numpy as jp
import jax.random as rax
import optax as optax
from flax import struct
from jax import config
from omegaconf import DictConfig, OmegaConf

import wandb
from coin_agent import GRUCoinAgent, CoinAgent
from three_player_coin_game import ThreePlayerCoinGame, episode_stats, coin_game_params, make_zero_episode, eval_agent_against_always_cooperate, eval_agent_against_always_defect
from utils import slurm_infos, rscope, global_norm, clip_grads_by_norm, npify, AliasDict, log_softmax_with_stop_grad_normalizing_constant


@struct.dataclass
class Optimizer:
    opt: Any = struct.field(pytree_node=False)
    opt_state: Any


@jax.jit
def tree_stack(xs):
    return jax.tree_map(lambda *args: jp.stack(args), *xs)


@partial(jax.jit, static_argnames=('B',))
def tree_unstack(xs, B):
    episodes = [jax.tree_map(lambda x: x[i], xs) for i in range(B)]
    return episodes


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


def use_rb(hp):
    return hp['agent_replay_buffer']['mode'] == 'enabled'


def set_up_state_from_config(hp):
    if hp['just_self_play']:
        just_self_play = True
        print('****just self play****')
    else:
        just_self_play = False

    dummy_env, _ = ThreePlayerCoinGame.init(
        rng=rax.PRNGKey(hp['seed']),
        **coin_game_params(hp),
    )
    dummy_episode = make_zero_episode(trace_length=hp['game']['game_length'], coin_game=dummy_env)

    if just_self_play:
        state = AliasDict(redirects={
            'agent1': 'agent0',
            'agent2': 'agent0',
            'rb_agent1_params': 'rb_agent0_params',
            'rb_agent2_params': 'rb_agent0_params',
        })
    else:
        state = dict()
    state['rng'] = rax.PRNGKey(hp['seed'])
    agent_module = GRUCoinAgent(hidden_size_actor=hp['actor']['hidden_size'],
                                hidden_size_qvalue=hp['qvalue']['hidden_size'],
                                layers_before_gru_actor=hp['actor']['layers_before_gru'],
                                layers_before_gru_qvalue=hp['qvalue']['layers_before_gru'], )
    dummy_rng = rax.PRNGKey(0)
    rng, rng1, rng2 = rax.split(state['rng'], 3)
    state['step'] = 0
    state['rng'] = rng
    dummy_obs_seq = dummy_episode['obs'][:, 0].reshape(dummy_episode['obs'].shape[0], -1)

    def set_up_agent_nn(player_id):
        agent_params = agent_module.init(rng1, {'obs_seq': dummy_obs_seq, 'rng': dummy_rng, 't': 0})
        agent = CoinAgent(params=agent_params, model=agent_module, player=player_id)
        state[f'agent{player_id}'] = agent

    set_up_agent_nn(player_id=0)
    agent0 = state['agent0']
    if not just_self_play:
        set_up_agent_nn(player_id=1)
        agent1 = state['agent1']
        set_up_agent_nn(player_id=2)
        agent2 = state['agent2']

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
            create_rb_agent_params(player_id=2)

    # --- defining ema ---
    state['agent0_ema'] = agent0
    if not just_self_play:
        state['agent1_ema'] = agent1
        state['agent2_ema'] = agent2

    # --- defining optimizers ---
    if hp['actor']['train']['optimizer'] == 'adam':
        actor_opt_module = optax.adam
    elif hp['actor']['train']['optimizer'] == 'sgd':
        actor_opt_module = optax.sgd
    else:
        raise ValueError(f"Unknown optimizer: {hp['actor']['train']['optimizer']}")

    actor_train_separate = hp['actor']['train']['separate']
    if actor_train_separate == 'enabled':
        actor_agent_lr = hp['actor']['train']['lr_loss_actor_agent']
        actor_opponent_lr = hp['actor']['train']['lr_loss_actor_opponent']
        actor_opt_agent = actor_opt_module(learning_rate=actor_agent_lr)
        actor_opt_opponent = actor_opt_module(learning_rate=actor_opponent_lr)

        def setup_actor_optimizer(player_id):
            agent = state[f'agent{player_id}']
            state[f'agent{player_id}_opt_actor_loss_agent'] = Optimizer(actor_opt_agent, actor_opt_agent.init(agent))
            state[f'agent{player_id}_opt_actor_loss_opponent'] = Optimizer(actor_opt_opponent, actor_opt_opponent.init(agent))

    elif actor_train_separate == 'disabled':
        lr = hp['actor']['train']['lr_loss_actor']
        actor_opt = actor_opt_module(learning_rate=lr)

        def setup_actor_optimizer(player_id):
            agent = state[f'agent{player_id}']
            state[f'agent{player_id}_opt_actor_loss'] = Optimizer(actor_opt, actor_opt.init(agent))
    else:
        raise ValueError(f"Unknown separate: {hp['actor']['train']['separate']}")

    setup_actor_optimizer(player_id=0)
    if not just_self_play:
        setup_actor_optimizer(player_id=1)
        setup_actor_optimizer(player_id=2)

    critic_lr = hp['qvalue']['train']['lr_loss_qvalue']
    if hp['qvalue']['train']['optimizer'] == 'adam':
        qvalue_opt = optax.adam(learning_rate=critic_lr)
    elif hp['qvalue']['train']['optimizer'] == 'sgd':
        qvalue_opt = optax.sgd(learning_rate=critic_lr)
    else:
        raise ValueError(f"Unknown optimizer: {hp['qvalue']['train']['optimizer']}")

    if hp['qvalue']['replay_buffer']['mode'] == 'disabled':
        pass
    else:
        raise ValueError(f'Unknown replay buffer mode: {hp["qvalue"]["replay_buffer"]["mode"]}')

    state['agent0_opt_qvalue'] = Optimizer(qvalue_opt, qvalue_opt.init(agent0))
    if not just_self_play:
        state['agent1_opt_qvalue'] = Optimizer(qvalue_opt, qvalue_opt.init(agent1))
        state['agent2_opt_qvalue'] = Optimizer(qvalue_opt, qvalue_opt.init(agent2))

    c_0 = agent0.get_initial_carries()
    c_0_actor = c_0['carry_actor']
    c_0_qvalue = c_0['carry_qvalue']

    if not just_self_play:
        c_1 = agent1.get_initial_carries()
        c_1_actor = c_1['carry_actor']
        c_1_qvalue = c_1['carry_qvalue']
        c_2 = agent2.get_initial_carries()
        c_2_actor = c_2['carry_actor']
        c_2_qvalue = c_2['carry_qvalue']
    else:
        c_1_actor = c_0_actor
        c_1_qvalue = c_0_qvalue
        c_2_actor = c_0_actor
        c_2_qvalue = c_0_qvalue

    carries = {'c_0_actor': c_0_actor,
               'c_0_qvalue': c_0_qvalue,
               'c_1_actor': c_1_actor,
               'c_1_qvalue': c_1_qvalue,
               'c_2_actor': c_2_actor,
               'c_2_qvalue': c_2_qvalue,
               }

    return state, carries


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


def train(hp, log_wandb):
    if log_wandb:
        run_id = wandb.run.id
    else:
        run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join(hp['save_dir'], run_id)
    os.makedirs(save_path, exist_ok=True)
    start_time = time.time()

    hp = flax.core.FrozenDict(hp)
    just_self_play = hp['just_self_play']
    state, carries = set_up_state_from_config(hp)
    # print all the keys in state
    print('state keys:', state.keys())
    # extract params for resetting in future
    agent0_params = state['agent0'].params
    agent1_params = state['agent1'].params
    agent2_params = state['agent2'].params

    dummy_env, _ = ThreePlayerCoinGame.init(
        rng=rax.PRNGKey(hp['seed']),
        **coin_game_params(hp),
    )
    episode_stats_jitted = jax.jit(lambda es: episode_stats(es, dummy_env))

    for i in range(500000):
        state['step'] = i
        state['rng'], rng = rax.split(state['rng'])
        # --- updating replay buffers ---
        if use_rb(hp) and i % hp['agent_replay_buffer']['update_freq'] == 0:
            def update_agent_rb(player_id: int):
                rb = state[f'rb_agent{player_id}_params']
                agent_params = state[f'agent{player_id}'].params
                rb = push_to_rb(rb, agent_params)
                state[f'rb_agent{player_id}_params'] = rb

            update_agent_rb(player_id=0)
            update_agent_rb(player_id=1)
            update_agent_rb(player_id=2)
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
                    'agent2s': sample_agents(player_id=2),
                }
            else:
                samples = {
                    'agent0s': sample_agents(player_id=0),
                    'agent1s': sample_agents(player_id=1),
                    'agent2s': sample_agents(player_id=2),
                }

        # --- generating episodes ---
        def gen_episode(player_id: int):
            if use_rb(hp):
                opponents = [samples[f'agent{(player_id + 1) % 3}s'], samples[f'agent{(player_id + 2) % 3}s']]
            else:
                opponents = [state[f'agent{(player_id + 1) % 3}'], state[f'agent{(player_id + 2) % 3}']]
            return generate_episodes(agent=state[f'agent{player_id}'],
                                     opponents=opponents,
                                     player_id=player_id,
                                     rng=rng,
                                     carries=carries,
                                     hp=hp)

        episodes = {}
        episodes[0] = gen_episode(player_id=0)
        if not just_self_play:
            episodes[1] = gen_episode(player_id=1)
            episodes[2] = gen_episode(player_id=2)

        # --- training actors ---
        def get_agent_update(player_id):
            if hp[f'agent_{player_id}'] == 'loqa':
                include_opponent = True
            elif hp[f'agent_{player_id}'] == 'naive':
                assert just_self_play is False, 'naive agent cannot be used in self-play'
                include_opponent = False
            else:
                raise ValueError(f'unknown agent type {hp[f"agent_{player_id}"]}')

            include_opponent = include_opponent and (False if i > hp['differentiable_opponent']['exclude_after_step'] else True)
            if i == hp['differentiable_opponent']['exclude_after_step']:
                print('excluding opponent from actor loss')

            if use_rb(hp):
                opponents = [samples[f'agent{(player_id + 1) % 3}s'], samples[f'agent{(player_id + 2) % 3}s']]
            else:
                opponents = [state[f'agent{(player_id + 1) % 3}'], state[f'agent{(player_id + 2) % 3}']]

            update = train_agent_actor(state=state,
                                       opponents=opponents,
                                       hp=hp,
                                       episodes=episodes[player_id],
                                       player_to_train=player_id,
                                       include_opponent=include_opponent)
            agent_entropy = -1 * update['loss_agent_entropy']
            grad_agent_norm = update['grad_agent_norm']
            grad_opponent_norm = update['grad_opponent_norm']
            return agent_entropy, grad_agent_norm, grad_opponent_norm

        agent0_entropy, grad_agent0_norm, grad_opponent0_norm = get_agent_update(player_id=0)
        if not just_self_play:
            agent1_entropy, grad_agent1_norm, grad_opponent1_norm = get_agent_update(player_id=1)
            agent2_entropy, grad_agent2_norm, grad_opponent2_norm = get_agent_update(player_id=2)

        # --- training qvalues ---
        def get_agent_qvalue_update(player_id):
            update = train_agent_qvalue(state, hp, episodes[player_id], player_to_train=player_id)
            qvalue_loss = update['qvalue_loss']
            return qvalue_loss

        loss_agent_0_qvalue = get_agent_qvalue_update(player_id=0)
        if not just_self_play:
            loss_agent_1_qvalue = get_agent_qvalue_update(player_id=1)
            loss_agent_2_qvalue = get_agent_qvalue_update(player_id=2)

        # ---- reset agents ----
        if hp['reset']['mode'] == 'disabled':
            pass
        elif hp['reset']['mode'] == 'enabled':
            if i % hp['reset']['every'] == 0:
                state['agent0'] = state['agent0'].replace(params=agent0_params)
                if not just_self_play:
                    state['agent1'] = state['agent1'].replace(params=agent1_params)
                    state['agent2'] = state['agent2'].replace(params=agent2_params)

        if i % hp['eval_every'] == 0:
            print(f'iteration {i}')
            print(f'loss_agent_0_qvalue: {loss_agent_0_qvalue.mean():.3f}')
            print(f'grad_agent0_norm: {grad_agent0_norm.mean():.3f}, grad_opponent0_norm: {grad_opponent0_norm.mean():.3f}')
            print(f'agent0_entropy: {agent0_entropy.mean():.3f}')
            if not just_self_play:
                print(f'loss_agent_1_qvalue: {loss_agent_1_qvalue.mean():.3f}')
                print(f'grad_agent1_norm: {grad_agent1_norm.mean():.3f}, grad_opponent1_norm: {grad_opponent1_norm.mean():.3f}')
                print(f'agent1_entropy: {agent1_entropy.mean():.3f}')
                print(f'loss_agent_2_qvalue: {loss_agent_2_qvalue.mean():.3f}')
                print(f'grad_agent2_norm: {grad_agent2_norm.mean():.3f}, grad_opponent2_norm: {grad_opponent2_norm.mean():.3f}')
                print(f'agent2_entropy: {agent2_entropy.mean():.3f}')

            episodes_eval = generate_eval_episodes(agent0=state['agent0'],
                                                   agent1=state['agent1'],
                                                   agent2=state['agent2'],
                                                   rng=rng,
                                                   carries=carries,
                                                   hp=hp)
            stats = episode_stats_jitted(episodes_eval)
            print(stats)

            if log_wandb:
                def flatten_stats(statistics):
                    ans = {}
                    # Loop through the original dictionary and extract values
                    for key, values in statistics.items():
                        ans[key + "_0"] = values[0]
                        ans[key + "_1"] = values[1]
                    return ans

                to_logs = {
                    'iteration': i,
                    'walltime': time.time() - start_time,
                    **flatten_stats(stats),
                }
                # Log the modified values using Weights & Biases (wandb)
                dict_for_agent0 = {
                    'loss_agent_0_qvalue': loss_agent_0_qvalue.mean(),
                    'agent0_entropy': agent0_entropy.mean(),
                    'grad_agent0_norm': grad_agent0_norm.mean(),
                    'grad_opponent0_norm': grad_opponent0_norm.mean(), }
                to_logs = {**to_logs, **dict_for_agent0}
                if not just_self_play:
                    dict_for_agent1 = {
                        'loss_agent_1_qvalue': loss_agent_1_qvalue.mean(),
                        'agent1_entropy': agent1_entropy.mean(),
                        'grad_agent1_norm': grad_agent1_norm.mean(),
                        'grad_opponent1_norm': grad_opponent1_norm.mean(), }
                    dict_for_agent2 = {
                        'loss_agent_2_qvalue': loss_agent_2_qvalue.mean(),
                        'agent2_entropy': agent2_entropy.mean(),
                        'grad_agent2_norm': grad_agent2_norm.mean(),
                        'grad_opponent2_norm': grad_opponent2_norm.mean(), }
                    to_logs = {**to_logs, **dict_for_agent1, **dict_for_agent2}

                wandb.log(to_logs, step=i)

                eval_agent_against_always_cooperate(state, hp, episode_stats_jitted, player=0)
                eval_agent_against_always_defect(state, hp, episode_stats_jitted, player=0)

        if i % hp['save_every'] == 0:
            # with open(os.path.join(save_path, f'state_{i}'), 'wb') as f:
            #     pickle.dump(flax.serialization.to_state_dict(state), f)
            minimal_state = {
                'agent0': npify(state['agent0']),
                'agent1': npify(state['agent1']),
                'hp': hp,
            }
            with open(os.path.join(save_path, f'minimal_state_{i}'), 'wb') as f:
                pickle.dump(flax.serialization.to_state_dict(minimal_state), f)


@jax.jit
def push_to_rb(rb, params):
    to_keep_rb = jax.tree_map(lambda x: x[1:], rb)
    params = jax.tree_map(lambda x: x[None], params)
    new_rb = jax.tree_map(lambda x, y: jp.concatenate((x, y), axis=0), to_keep_rb, params)
    return new_rb


@partial(jax.jit, static_argnames=('hp',))
def generate_eval_episodes(agent0, agent1, agent2, carries, rng, hp):
    rngs = rax.split(rscope(rng, f'gen_episode_eval'), hp['batch_size'])
    rngs = jp.stack(rngs, axis=0)
    init_envs, _ = jax.vmap(lambda r: ThreePlayerCoinGame.init(
        rng=r,
        **coin_game_params(hp),
    ))(rscope(rngs, "game_init"))

    episodes_eval, aux = jax.vmap(lambda r, env:
                                  play_episode_scan_inner_gru(dict(agent0=agent0,
                                                                   agent1=agent1,
                                                                   agent2=agent2,
                                                                   rng=r,
                                                                   t=0,
                                                                   **carries),
                                                              trace_length=hp['game']['game_length'],
                                                              env=env),
                                  )(rscope(rngs, "play_rng"), init_envs)
    return episodes_eval


@partial(jax.jit, static_argnames=('hp', 'player_id'))
def generate_episodes(agent, opponents, player_id: int, rng, carries, hp):
    rngs = rax.split(rscope(rng, f'gen_episode_{player_id}'), hp['batch_size'])
    rngs = jp.stack(rngs, axis=0)
    init_envs, _ = jax.vmap(lambda r: ThreePlayerCoinGame.init(
        rng=r,
        **coin_game_params(hp),
    ))(rscope(rngs, "game_init"))

    def func(r, env, op):
        return play_episode_scan_inner_gru(dict(**{f'agent{player_id}': agent,
                                                   f'agent{(player_id + 1) % 3}': op[0],
                                                   f'agent{(player_id + 2) % 3}': op[1]},  # extend to 3 players
                                                rng=r,
                                                t=0,
                                                **carries),
                                           trace_length=hp['game']['game_length'],
                                           env=env)

    if use_rb(hp):
        episodes, aux = jax.vmap(func)(rscope(rngs, "play_rng"), init_envs, opponents)
    else:
        opponent = opponents[0]
        episodes, aux = jax.vmap(lambda r, env: func(r, env, opponent))(rscope(rngs, "play_rng"), init_envs)

    return episodes


@partial(jax.jit, static_argnames=['trace_length'])
def play_episode_scan_inner_gru(inp, trace_length, env):
    agent0 = inp['agent0']
    agent1 = inp['agent1']
    agent2 = inp['agent2']
    rng = inp['rng']
    c_0_actor = inp['c_0_actor']
    c_0_qvalue = inp['c_0_qvalue']
    c_1_actor = inp['c_1_actor']
    c_1_qvalue = inp['c_1_qvalue']
    c_2_actor = inp['c_2_actor']
    c_2_qvalue = inp['c_2_qvalue']

    episode = make_zero_episode(trace_length=trace_length, coin_game=env)

    # set initial observations
    episode["obs"] = episode["obs"].at[0].set(env.get_obs())
    episode["coin_pos"] = episode["coin_pos"].at[0].set(env.coin_pos)
    episode["coin_owner"] = episode["coin_owner"].at[0].set(env.coin_owner)
    episode["player1_pos"] = episode["player1_pos"].at[0].set(env.players_pos[0])
    episode["player2_pos"] = episode["player2_pos"].at[0].set(env.players_pos[1])
    episode["player3_pos"] = episode["player3_pos"].at[0].set(env.players_pos[2])

    def body_fn(carry, _):
        aux = {}
        env, rng, episode, t, c_0_actor, c_0_qvalue, c_1_actor, c_1_qvalue, c_2_actor, c_2_qvalue = carry
        rng, rng0, rng1, rng2 = rax.split(rng, 4)
        episode['games'] = jax.tree_map(lambda x, o: x.at[t].set(o), episode['games'], env)
        prev_obs_0 = episode['obs'][t, 0].reshape(-1)
        prev_obs_1 = episode['obs'][t, 1].reshape(-1)
        prev_obs_2 = episode['obs'][t, 2].reshape(-1)
        out1 = agent0.call_step({'obs': prev_obs_0, 'rng': rng0, 't': t, 'carry_actor': c_0_actor, 'carry_qvalue': c_0_qvalue})
        out2 = agent1.call_step({'obs': prev_obs_1, 'rng': rng1, 't': t, 'carry_actor': c_1_actor, 'carry_qvalue': c_1_qvalue})
        out3 = agent2.call_step({'obs': prev_obs_2, 'rng': rng2, 't': t, 'carry_actor': c_2_actor, 'carry_qvalue': c_2_qvalue})
        action1 = out1['action']
        action2 = out2['action']
        action3 = out3['action']
        logp1 = out1['logp']
        logp2 = out2['logp']
        logp3 = out3['logp']
        c_0_actor = out1['carry_actor']
        c_1_actor = out2['carry_actor']
        c_0_qvalue = out1['carry_qvalue']
        c_1_qvalue = out2['carry_qvalue']
        c_2_actor = out3['carry_actor']
        c_2_qvalue = out3['carry_qvalue']
        act = jp.stack([action1, action2, action3])
        logp = jp.stack([logp1, logp2, logp3], axis=0)
        assert act.shape == (3,)
        env, obs, rew = env.step(act)
        episode["obs"] = episode["obs"].at[1 + t].set(obs)
        episode["coin_pos"] = episode["coin_pos"].at[1 + t].set(env.coin_pos)
        episode["coin_owner"] = episode["coin_owner"].at[1 + t].set(env.coin_owner)
        episode["player1_pos"] = episode["player1_pos"].at[1 + t].set(env.players_pos[0])
        episode["player2_pos"] = episode["player2_pos"].at[1 + t].set(env.players_pos[1])
        episode["player3_pos"] = episode["player3_pos"].at[1 + t].set(env.players_pos[2])
        episode["act"] = episode["act"].at[t].set(act)
        episode["logp"] = episode["logp"].at[t].set(logp)
        episode["rew"] = episode["rew"].at[t].set(rew)
        return (env, rng, episode, t + 1, c_0_actor, c_0_qvalue, c_1_actor, c_1_qvalue, c_2_actor, c_2_qvalue), aux

    (env, rng, episode, _, _, _, _, _, _, _), aux = jax.lax.scan(f=body_fn, init=(env, rng, episode, 0, c_0_actor, c_0_qvalue, c_1_actor, c_1_qvalue, c_2_actor, c_2_qvalue), xs=(), length=trace_length)
    last_game = env
    episode['games'] = jax.tree_map(lambda x, o: x.at[trace_length].set(o), episode['games'], last_game)
    return episode, aux


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
    agent = state[f'agent{player_to_train}']
    agent_ema = state[f'agent{player_to_train}_ema']
    agent_opt = state[f'agent{player_to_train}_opt_qvalue']
    aux = update_agent_qvalue(agent, agent_ema, agent_opt, hp, episodes, player_to_train)
    new_agent = aux['agent']
    state[f'agent{player_to_train}'] = new_agent
    ema_gamma = hp['qvalue']['train']['target_ema_gamma']
    agent_ema_params = jax.tree_map(lambda old, new: ema_gamma * old + (1 - ema_gamma) * new, agent_ema.params, new_agent.params)
    state[f'agent{player_to_train}_ema'] = agent_ema.replace(params=agent_ema_params)
    state[f'agent{player_to_train}_opt_qvalue'] = agent_opt.replace(opt_state=aux['new_opt_state'])
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


def agent_opponent_loss(episodes, opponent, opponent_player, hp, all_values, returns, agent_rewards, logps, op_rewards_baseline, advantage):
    T = episodes['obs'].shape[0] - 1

    op_rewards = episodes['rew'][:, opponent_player]  # [T]
    op_actions = episodes['act'][:, opponent_player]  # [T]

    opponent_obs = episodes['obs'][:, opponent_player].reshape(T + 1, -1)
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

    elif hp['differentiable_opponent']['method'] in ['loaded-dice', 'loaded-ios', 'loaded-aa']:

        if hp['differentiable_opponent']['method'] == 'loaded-dice':
            op_returns = differentiable_loaded_dice_returns(op_rewards=op_rewards, agent_logps=logps, op_values=op_values, hp=hp)

            if hp['differentiable_opponent']['differentiable_current_reward']:
                op_differentiable_qvalues = op_returns
            else:
                op_differentiable_qvalues = op_rewards[:-1] + hp['reward_discount'] * op_returns[1:]
                op_differentiable_qvalues = jp.concatenate([op_differentiable_qvalues, op_rewards[-1:]])

            timesteps = jp.arange(op_differentiable_qvalues.shape[0])  # [T]
            dif_op_qvalues = op_qvalues[:-1].at[timesteps, op_actions].set(op_differentiable_qvalues)  # [T, 4]

            if hp['differentiable_opponent']['stop_grad_normalizing_constant'] == 'enabled':
                op_logps = log_softmax_with_stop_grad_normalizing_constant(dif_op_qvalues / hp['op_softmax_temp'])
            elif hp['differentiable_opponent']['stop_grad_normalizing_constant'] == 'disabled':
                op_logps = jax.nn.log_softmax(dif_op_qvalues / hp['op_softmax_temp'])  # [T, 4]
            else:
                raise ValueError(f"Unknown stop_grad_normalizing_constant {hp['differentiable_opponent']['stop_grad_normalizing_constant']}")

            op_logps = jax.vmap(lambda lps, act: lps[act])(op_logps, op_actions)  # [T]
            loss_opponent = -(op_logps * advantage).mean()

        elif hp['differentiable_opponent']['method'] == 'loaded-ios':
            print("loaded-ios")
            inf_weight = hp['actor']['inf_weight']
            op_returns = differentiable_loaded_dice_returns(op_rewards=op_rewards, agent_logps=logps, op_values=op_values, hp=hp)
            # op_advantage = op_rewards + hp['reward_discount'] * op_values[1:] - op_values[:-1]
            op_advantage = op_returns - op_values[:-1]

            batch_advantage = jax.lax.all_gather(advantage, axis_name='batch')
            n_advantage = (advantage - jp.mean(batch_advantage)) / jp.std(batch_advantage)

            batch_op_advantage = jax.lax.all_gather(op_advantage, axis_name='batch')
            n_op_advantage = (op_advantage - jp.mean(batch_op_advantage)) / jp.std(batch_op_advantage)
            mask = -1 * jp.where(op_advantage > 0, 1., 0.) * jp.where(advantage > 0, 1., 0.)
            mask = mask - 1 * jp.where(op_advantage > 0, 1., 0.) * jp.where(advantage < 0, 1., 0.)
            loss_opponent = (op_advantage * advantage * mask).mean()

        elif hp['differentiable_opponent']['method'] == 'loaded-aa':
            # import pdb; pdb.set_trace()
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
            loss_opponent = -(op_differentiable_qvalues * advantage).mean()

    else:
        raise ValueError(f"Unknown differentiable opponent method {hp['differentiable_opponent']['method']}")
    return loss_opponent


@partial(jax.jit, static_argnames='hp')
def agent_policy_loss(agent, opponent, hp, episodes, player_to_train: int, op_1_rewards_baseline, op_2_rewards_baseline):
    # simple reinforce with baseline
    agent_rewards = episodes['rew'][:, player_to_train]
    returns = get_returns(agent_rewards, hp)  # [T]

    dummy_rng = rax.PRNGKey(0)
    T = episodes['obs'].shape[0] - 1
    player_obs = episodes['obs'][:, player_to_train].reshape(T + 1, -1)
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

    op_1 = opponent[0]
    op_2 = opponent[1]
    loss_opponent_1 = agent_opponent_loss(episodes, op_1, (player_to_train + 1) % 3, hp, all_values, returns, agent_rewards, logps, op_1_rewards_baseline, advantage)
    loss_opponent_2 = agent_opponent_loss(episodes, op_2, (player_to_train + 2) % 3, hp, all_values, returns, agent_rewards, logps, op_1_rewards_baseline, advantage)
    loss_opponent = loss_opponent_1 + loss_opponent_2

    return {'loss_agent': loss_agent,
            'loss_opponent': loss_opponent,
            'loss_agent_entropy': loss_agent_entropy}


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


def magic_box(z):
    return jp.exp(z - jax.lax.stop_gradient(z))


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


@partial(jax.jit, static_argnames=('hp', 'include_opponent'))
def update_agent_actor(agent, opponents, optimizers, hp, episodes, player_to_train: int, include_opponent: bool = False):
    op_1_rewards_baseline = episodes['rew'][..., (player_to_train + 1) % 3].mean()
    op_2_rewards_baseline = episodes['rew'][..., (player_to_train + 2) % 3].mean()

    def loss_fn(a):
        def func(ep, op):
            return agent_policy_loss(agent=a,
                                     opponent=op,
                                     hp=hp,
                                     episodes=ep,
                                     player_to_train=player_to_train,
                                     op_1_rewards_baseline=op_1_rewards_baseline,
                                     op_2_rewards_baseline=op_2_rewards_baseline
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

    if hp['actor']['train']['separate'] == 'disabled':
        opt_loss = optimizers['opt_loss']
        if include_opponent:
            grad = jax.tree_map(lambda a, b: a + hp['opponent_differentiation_weight'] * b, grad_agent, grad_opponent)
        else:
            grad = grad_agent

        grad = clip_grad(grad)
        updates, new_opt_loss_state = opt_loss.opt.update(grad, opt_loss.opt_state, agent)
        new_agent = optax.apply_updates(agent, updates)
        new_optimizer_states = {'new_opt_loss_state': new_opt_loss_state}

    elif hp['actor']['train']['separate'] == 'enabled':
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

    if hp['actor']['train']['separate'] == 'enabled':
        opt_loss_agent = state[f'agent{player_to_train}_opt_actor_loss_agent']
        opt_loss_opponent = state[f'agent{player_to_train}_opt_actor_loss_opponent']
        optimizers = {'opt_loss_agent': opt_loss_agent, 'opt_loss_opponent': opt_loss_opponent}
    elif hp['actor']['train']['separate'] == 'disabled':
        opt_loss = state[f'agent{player_to_train}_opt_actor_loss']
        optimizers = {'opt_loss': opt_loss}
    else:
        raise ValueError(f"Unknown actor separate mode {hp['actor']['train']['actor']['separate']}")

    aux = update_agent_actor(agent=agent,
                             opponents=opponents,
                             optimizers=optimizers,
                             hp=hp,
                             episodes=episodes,
                             player_to_train=player_to_train,
                             include_opponent=include_opponent, )
    if hp['actor']['train']['separate'] == 'enabled':
        state[f'agent{player_to_train}_opt_actor_loss_agent'] = opt_loss_agent.replace(opt_state=aux['new_opt_loss_agent_state'])
        state[f'agent{player_to_train}_opt_actor_loss_opponent'] = opt_loss_opponent.replace(opt_state=aux['new_opt_loss_opponent_state'])
    elif hp['actor']['train']['separate'] == 'disabled':
        state[f'agent{player_to_train}_opt_actor_loss'] = opt_loss.replace(opt_state=aux['new_opt_loss_state'])
    state[f'agent{player_to_train}'] = aux['agent']
    return aux


def entropy(logps):
    return -jp.sum(jp.nan_to_num(logps * jp.exp(logps)), axis=-1)


def get_returns(rewards, hp):
    returns = discounted_returns(rewards, hp['reward_discount'])
    return returns


def run_tests():
    pass


@hydra.main(version_base=None, config_path="conf/coin_conf", config_name="coin_config")
def main(cfg: DictConfig) -> None:
    jp.set_printoptions(precision=3)
    if cfg.hp.differentiable_opponent.method == 'n_step':
        assert cfg.hp.differentiable_opponent.n_step != 1, 'n_step=1 breaks the logic of the code, especially the slicing [:(-1+1)] returns [] where we want the whole array'

    config.update('jax_disable_jit', cfg.jax.jax_disable_jit)
    config.update("jax_debug_nans", cfg.jax.jax_debug_nans)
    hp = OmegaConf.to_container(cfg.hp, resolve=True)  # Converts cfg to a Python dict
    print(OmegaConf.to_yaml(cfg.hp))  # Prints the hyperparameters

    log_wandb = cfg.wandb.state == 'enabled'
    if log_wandb:
        wandb_id = wandb.util.generate_id()
        wandb.init(project="loqa-ipd", id=wandb_id, dir=cfg.wandb.wandb_dir, tags=cfg.wandb.tags)
        wandb.config.update(hp)
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

        # go recursively to ./conf and its subdirectories and save every file with yaml
        for root, dirs, files in os.walk("conf"):
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
