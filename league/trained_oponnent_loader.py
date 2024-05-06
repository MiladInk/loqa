import dataclasses
from functools import partial

import flax.struct as struct
import optax
import os
# add agent.py to the path
import sys

import pandas as pd
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from league.agent import get_cooperative_action, ConvAgent, init_agents
from league.coin import CoinGame, play_episode_scan, episode_stats
from league._utils import rscope, value_loss, policy_loss, clip_grads_by_norm, Optimizer, rlax_entropy_loss
import jax
import jax.numpy as jp
import jax.random as rax


@partial(jax.jit, static_argnames=('hp', 'batch_size'))
def generate_episodes(agent, opponent, hp, coin_game_template, rng, batch_size):
    def get_actions(subrng, env, episode, t):
        subrngs = jax.random.split(subrng, 2)

        agent_1_action = agent.get_action(rng=subrngs[0],
                                          episode=episode,
                                          t=t,
                                          hp=hp,
                                          coin_game_template=coin_game_template,
                                          oponnent=None,
                                          env=env)

        agent_2_action = opponent.get_action(rng=subrngs[1],
                                             episode=episode,
                                             t=t,
                                             hp=hp,
                                             coin_game_template=coin_game_template,
                                             oponnent=None,
                                             env=env)

        actions = jax.lax.select(agent.player == 0, jp.stack([agent_1_action, agent_2_action]), jp.stack([agent_2_action, agent_1_action]))
        return actions, ()

    rngs = rax.split(rng, batch_size)
    init_envs, _ = jax.vmap(lambda r: CoinGame.init(r,
                                                    height=hp['height'],
                                                    width=hp['width'],
                                                    gnumactions=hp['g_num_actions'],
                                                    trace_length=hp['trace_length'],
                                                    new_coin_every_turn=hp['new_coin_every_turn'],
                                                    dtype=hp['dtype']))(rscope(rngs, "game_init"))

    episodes, _ = (jax.vmap(lambda e, r: play_episode_scan(e, get_actions, r, trace_length=hp['trace_length']))(init_envs, rscope(rngs, "game_play")))

    return episodes

def get_value_loss(agent, target_agent, episodes, hp):
    def loss_single_episode(ep):
        agent_values = agent.get_values(episode=ep)
        target_values = target_agent.get_values(episode=ep)
        rewards = ep['rew'][:, agent.player]
        assert agent_values.shape == (rewards.shape[0] + 1,)

        return value_loss(rewards=rewards,
                          target_values=target_values,
                          values=agent_values,
                          hp=hp)

    return jax.vmap(loss_single_episode)(episodes).mean()

def get_episode_agent_logits(agent, episode):
    all_hiddens = agent.observe(episode=episode)
    # remove last hidden state, which is not used
    all_hiddens = all_hiddens[:-1, ...]
    # jax.debug.print('all_hiddens shape = {s}', s=all_hiddens.shape)
    return jax.vmap(lambda h: agent.logits(hiddens=h))(all_hiddens)

def agent_loss(agent, episodes, hp):
    def loss_single_episode(ep):
        agent_logits = get_episode_agent_logits(agent=agent, episode=ep)

        actions = ep['act'][:, agent.player]
        agent_logps = jax.vmap(lambda l, a: jax.nn.log_softmax(l)[a])(agent_logits, actions)

        entropy_loss = rlax_entropy_loss(logits_t=agent_logits, w_t=jp.ones_like(agent_logps))  # [1]

        values = agent.get_values(episode=ep)
        rewards = ep['rew'][:, agent.player]
        pg_loss_agent = policy_loss(logps=agent_logps, values=values, rewards=rewards, hp=hp)

        pg_loss = pg_loss_agent
        return {
            'entropy_loss': entropy_loss,
            'policy_gradient_loss_no_discount': pg_loss,
            'pg_loss_agent': pg_loss_agent,
        }  # [1]

    aux = jax.vmap(loss_single_episode)(episodes)
    return jax.tree_map(lambda x: x.mean(), aux)


@partial(jax.jit, static_argnames=('hp', 'do_ppo'))
def update_agent_actor(episodes, agent, old_agent, agent_opt_actor, hp, do_ppo=False):
    def pg_loss_agent(a):
        aux = agent_loss(episodes=episodes, agent=a, hp=hp)
        loss = aux['pg_loss_agent'] + hp['agent_entropy_beta'] * aux['entropy_loss']
        return loss, aux

    pg_grad, pg_loss_aux = jax.grad(pg_loss_agent, has_aux=True)(agent)

    pg_loss = pg_loss_aux['policy_gradient_loss_no_discount']
    entropy_loss = pg_loss_aux['entropy_loss']
    pg_grad_before_clip = pg_grad
    pg_grad = clip_grads_by_norm(pg_grad, hp['max_grad_norm'])
    updates, new_agent_opt_actor_state = agent_opt_actor.opt.update(pg_grad, agent_opt_actor.opt_state, agent)
    new_agent = optax.apply_updates(params=agent, updates=updates)

    update = {
        'new_agent': new_agent,
        'new_agent_opt_actor_state': new_agent_opt_actor_state,
        'entropy_loss': entropy_loss,
        'policy_gradient_loss_no_discount': pg_loss,
        'grad_before_clip': pg_grad_before_clip,
        'grad_after_clip': pg_grad,
    }
    return update

@partial(jax.jit, static_argnames=('hp',))
def update_agent_value(episodes, agent, agent_opt_value, target_agent, hp: 'Hp'):
    def loss(a):
        return get_value_loss(episodes=episodes, agent=a, target_agent=target_agent, hp=hp)

    v_loss, v_grad = jax.value_and_grad(loss)(agent)

    v_grad_before_clip = v_grad
    v_grad = clip_grads_by_norm(v_grad, hp['max_grad_norm'])
    updates, new_agent_opt_value_state = agent_opt_value.opt.update(v_grad, agent_opt_value.opt_state, agent)
    new_agent = optax.apply_updates(params=agent, updates=updates)

    return {'new_agent': new_agent,
            'new_agent_opt_value_state': new_agent_opt_value_state,
            'agent_value_function_loss': v_loss,
            'grad_before_clip': v_grad_before_clip,
            'grad_after_clip': v_grad,
            }

def train_gru_vs_opponent(opponent: object, hp: object, coin_game_template: object, rng: object, batch_size: object, do_ppo: bool = False) -> object:
    agent_player = 1 - opponent.player
    print(f'agent player: {agent_player}, opponent player: {opponent.player}')

    opponent_module = ConvAgent(coin_game=coin_game_template,
                                conv_aggregator_activation='relu',
                                preprocess_obs_config={'mode': 'raw_flat'},
                                value_mlp_features=[1],
                                actor_mlp_features=[hp['g_num_actions']],
                                horizon=-1,
                                use_film_in_value_for_time=False,
                                film_size=32,
                                aggregator_mlp_features=[32, 32],
                                aggregator_type='pola_gru_2',
                                player=agent_player,
                                )
    agent = init_agents([opponent_module], rng, coin_game_template, hp['trace_length'])[0]

    opt_actor = optax.adam(learning_rate=3e-4)
    opt_value = optax.adam(learning_rate=3e-4)

    agent_opt_actor = Optimizer(opt=opt_actor, opt_state=opt_actor.init(agent))
    agent_opt_value = Optimizer(opt=opt_value, opt_state=opt_value.init(agent))

    odds_own_coin = []
    iterations = []
    for i in range(3000):
        rng, rng0 = rax.split(rng, 2)
        episodes = generate_episodes(agent=agent, opponent=opponent, hp=hp, coin_game_template=coin_game_template, rng=rng0, batch_size=batch_size)

        old_agent = agent
        for _ in range(4 if do_ppo else 1):
            update_actor = update_agent_actor(episodes=episodes, agent=agent, old_agent=old_agent, agent_opt_actor=agent_opt_actor, hp=hp, do_ppo=do_ppo)
            agent = update_actor['new_agent']
            agent_opt_actor = agent_opt_actor.replace(opt_state=update_actor['new_agent_opt_actor_state'])

        update_value = update_agent_value(episodes=episodes, agent=agent, agent_opt_value=agent_opt_value, target_agent=agent, hp=hp)
        agent = update_value['new_agent']
        agent_opt_value = agent_opt_value.replace(opt_state=update_value['new_agent_opt_value_state'])

        stats = episode_stats(episodes, coin_game_template)

        info = {'mean_rewards_agent': stats["mean_rewards"][agent_player],
                'mean_rewards_oponnent': stats["mean_rewards"][opponent.player],}

        if i % 100 == 0:
            print(f'i:{i}')
            stats = episode_stats(episodes, coin_game_template)
            odd = 1. - stats['adversarial_pickup_div_all_pickup'][opponent.player]
            odds_own_coin.append(odd)
            iterations.append(i)
            print(f'Loss: {update_actor["policy_gradient_loss_no_discount"]:.2f}, '
                  f'entropy: {update_actor["entropy_loss"]:.2f}, '
                  f'v_loss: {update_value["agent_value_function_loss"]:.2f}',)
            print(f'Agent: {info["mean_rewards_agent"]:.2f}, '
                  f'Opponent: {info["mean_rewards_oponnent"]:.2f}')

        # plot and save the plot of the odds of picking own coin at the end

        # plt.plot(iterations, odds_own_coin, color='blue', linewidth=1.0)
        # # add labels
        # plt.xlabel('Iterations')
        # plt.ylabel('Trained Opponent\'s Odds of Own Coin')
        # plt.legend(['Trained Opponent vs. BRS-SP'])
        #
        # # save the plot
        # plt.savefig(f'trained_opponent_vs_brs-sp.png')
        #
        # # save the iterations and odds of picking own coin at the end in a csv file
        # df = pd.DataFrame({'iterations': iterations, 'odds_own_coin': odds_own_coin})
        # df.to_csv('trained_opponent_vs_brs-sp.csv', index=False)

    return agent


@dataclasses.dataclass
class TrainedAgentLoader:
  hp: dict
  supported_players: list = dataclasses.field(default_factory=lambda: [1])
  name: str = 'Trained Agent'

  def load(self, player, *args, **kwargs):
    assert player in self.supported_players
    return train_gru_vs_opponent(opponent=kwargs.get('agent'),
                                 hp=self.hp,
                                 coin_game_template=kwargs.get('coin_game_template'),
                                 rng=kwargs.get('rng'),
                                 batch_size=kwargs.get('batch_size'),
                                 do_ppo=False,)
