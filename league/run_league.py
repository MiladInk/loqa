import os
import sys
from pathlib import Path

import fire
import flax.core
import jax
import jax.numpy as jp
import jax.random as rax
import multiprocessing as mp

from tqdm import tqdm



# Construct the relative path
current_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.dirname(current_directory)
sys.path.append(project_directory)
sys.path.append(current_directory)

from league.random_wrapper import RandomAgentLoader
from league.loqa_agent_loader import LOQAAgentLoader
from league.advantage_alignment_agent_loader import AdvantageAlignmentAgentLoader
from always_cooperate_wrapper import AlwaysCooperateAgentLoader
from always_defect_wrapper import AlwaysDefectAgentLoader
from league.coin import CoinGame, play_episode_unroll, episode_stats
from pola_agent_loader import POLAAgentLoader
from league._utils import rscope, npify
from league.mfos_wrapper import MFOSAgentLoader

def eval_agents(agent1, agent2, batch_size: int, rng: rax.PRNGKey, hp, coin_game_template, for_instead_of_vmap=False):
    assert agent1.player == 0
    assert agent2.player == 1

    # dirty workaround to make multiprocessing work https://stackoverflow.com/questions/67023124/workaround-for-multiprocessing-with-local-functions-in-python
    def get_actions(subrng, env, episode, t):
        subrngs = jax.random.split(subrng, 2)

        agent_1_action = agent1.get_action(rng=subrngs[0],
                                           episode=episode,
                                           t=t,
                                           hp=hp,
                                           coin_game_template=coin_game_template,
                                           oponnent=agent2,
                                           env=env)

        agent_2_action = agent2.get_action(rng=subrngs[1],
                                           episode=episode,
                                           t=t,
                                           hp=hp,
                                           coin_game_template=coin_game_template,
                                           oponnent=agent1,
                                           env=env)

        return jp.stack([agent_1_action, agent_2_action]), ()

    rngs = rax.split(rng, batch_size)
    init_envs, _ = jax.vmap(lambda r: CoinGame.init(r,
                                                    height=hp['height'],
                                                    width=hp['width'],
                                                    gnumactions=hp['g_num_actions'],
                                                    trace_length=hp['trace_length'],
                                                    new_coin_every_turn=hp['new_coin_every_turn'],
                                                    dtype=hp['dtype']))(rscope(rngs, "game_init"))
    if for_instead_of_vmap is False:
        episodes, _ = (jax.vmap(lambda e, r: play_episode_unroll(e, get_actions, r, trace_length=hp['trace_length']))(init_envs, rscope(rngs, "game_play")))
    else: # just because m-fos is a torch module and we cannot vmap over it
        episodes = []
        for i, rng in enumerate(rscope(rngs, "game_play")):
            init_env = jax.tree_map(lambda x: x[i], init_envs)
            episode, _ = play_episode_unroll(init_env, get_actions, rng, trace_length=hp['trace_length'])
            episodes.append(episode)
        episodes = jax.tree_map(lambda *x: jp.stack(x), *episodes)

    return episodes


def coin_game_params(hp):
    ans = {'height': hp['height'],
           'width': hp['width'],
           'gnumactions': hp['g_num_actions'],
           'trace_length': hp['trace_length'],
           'new_coin_every_turn': hp['new_coin_every_turn'],
           'dtype': hp['dtype'],
           }
    return ans


def get_coin_game_template(hp) -> CoinGame:
    coin_game_template, _ = CoinGame.init(rng=rax.PRNGKey(0), **coin_game_params(hp))
    return coin_game_template


def evaluate_these_agent_combinations(combinations, batch_size, rng, hp, for_instead_of_vmap=False):
    results = {}
    episodes_logs = {}
    coin_game_template = get_coin_game_template(hp)  # this is important, as all our agents should use the same for jit cache not to cause error
    for agent1_loader, agent2_loader in combinations:
        if (0 not in agent1_loader.supported_players) or (1 not in agent2_loader.supported_players):
            print(f"Skipping {agent1_loader.name} vs {agent2_loader.name} because of unsupported players")
            continue

        assert hp['width'] == hp['height'], "width and height must be equal as mfos just accepts a grid size not separate width and height"
        agent1 = agent1_loader.load(player=0, coin_game_template=coin_game_template, grid_size=hp['width'])
        agent2 = agent2_loader.load(player=1, coin_game_template=coin_game_template, agent=agent1, rng=rng, batch_size=batch_size, grid_size=hp['width'])

        key = (agent1_loader.name, agent2_loader.name)

        print(f"{key[0]} vs {key[1]}:")

        if agent1.player != 0 or agent2.player != 1:
            continue

        if 'mfos' in agent1_loader.name.lower() or 'mfos' in agent2_loader.name.lower() or for_instead_of_vmap is True:
            for_instead_of_vmap = True
        else:
            for_instead_of_vmap = False
        episodes = eval_agents(agent1, agent2, batch_size, rng, hp, coin_game_template, for_instead_of_vmap=for_instead_of_vmap)
        stats = episode_stats(episodes, coin_game_template)

        if key not in results:
            results[key] = []
        results[key].append(stats)

        if key not in episodes_logs:
            episodes_logs[key] = []
        episodes_logs[key].append(episodes)

        print(stats)
        print()

    return results, episodes_logs


class DotDict(dict):
    def __getattr__(self, name):
        # Called when an attribute isn't found in the usual places (e.g. instance attribute or in its class)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        # This allows d.x = y to set d['x'] = y
        self[name] = value

    def __delattr__(self, name):
        # This allows del d.x to delete d['x']
        if name in self:
            del self[name]
        else:
            raise AttributeError(name)


def get_all_agent_loaders(hp):
    import os
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    loqa_agent_loaders = [LOQAAgentLoader(path=os.path.join(current_dir, 'checkpoints/loqa_ckpt_10_seeds_v1/', run_id, 'minimal_state_6000') ,name=f'loqa_{run_id}')
                          for run_id in ['2i4vsulp', '2ynm2tt3', '2zgb3k1q','3islgbi7','3r9977xr','20htxiyc','30j2wrur','31zh1lui','37u1bq3f','233m1v5f']]

    loqa_rb_ablation_loader = [LOQAAgentLoader(path=os.path.join(current_dir, 'checkpoints/loqa_ckpt_10_seeds_v1_rb_ablation/', run_id, 'minimal_state_6000') ,name=f'loqa_rb_ablation_{run_id}')
                          for run_id in ['1abkzsx3','1tieobkn', '2xndvhhp', '2zpspi7d', '3kfuinhg', '3o6xocrm', '3tocppgh', '375wzono', 'd3g7n3wp', 'ejpsvuwk']]

    loqa_sp_ablation_loader = [LOQAAgentLoader(path=os.path.join(current_dir, 'checkpoints/loqa_ckpt_10_seeds_v1_sp_ablation/', run_id, 'minimal_state_6000') ,name=f'loqa_sp_ablation_{run_id}')
                          for run_id in ['1ecqpbed', '2ntyuzde', '3cfm0yd6', '3cfulper', '16qp3f50', '20gama5k', '60m01f0h', 'aehjqk0o', 'dqoxc4wd', 'q9u2wcx2']]

    advantage_alignment_loaders = [AdvantageAlignmentAgentLoader(path=os.path.join(current_dir, 'checkpoints/advantage_alignment_seeds/', f'minimal_state_25400_{seed}'), name=f'advantage_alignment_{seed}') for seed in range(1,11)]

    advantage_alignment_mask_cooperative_empathetic = [AdvantageAlignmentAgentLoader(path=os.path.join(current_dir, 'checkpoints/advantage_alignment_seeds/', f'mask_cooperative_empathetic_minimal_state_{seed}'), name=f'advantage_alignment_mask_cooperative_empathetic_{seed}') for seed in range(1,10)]
    advantage_alignment_mask_empathetic = [AdvantageAlignmentAgentLoader(path=os.path.join(current_dir, 'checkpoints/advantage_alignment_seeds/', f'mask_empathetic_minimal_state_{seed}'), name=f'advantage_alignment_mask_empathetic_{seed}') for seed in range(1,10)]
    advantage_alignment_mask_spiteful = [AdvantageAlignmentAgentLoader(path=os.path.join(current_dir, 'checkpoints/advantage_alignment_seeds/', f'mask_spiteful_minimal_state_{seed}'), name=f'advantage_alignment_mask_spiteful_{seed}') for seed in range(1,10)]
    advantage_alignment_mask_vengeful = [AdvantageAlignmentAgentLoader(path=os.path.join(current_dir, 'checkpoints/advantage_alignment_seeds/', f'mask_vengeful_minimal_state_{seed}'), name=f'advantage_alignment_mask_vengeful_{seed}') for seed in range(1,10)]
    advantage_alignment_mask_vengeful_spiteful = [AdvantageAlignmentAgentLoader(path=os.path.join(current_dir, 'checkpoints/advantage_alignment_seeds/', f'mask_vengeful_spiteful_minimal_state_{seed}'), name=f'advantage_alignment_mask_vengeful_spiteful_{seed}') for seed in range(1,10)]
    advantage_alignment_new_baseline = [AdvantageAlignmentAgentLoader(path=os.path.join(current_dir, 'checkpoints/advantage_alignment_seeds/', f'aa_baseline_minimal_state_{seed}'), name=f'advantage_alignment_new_baseline_{seed}') for seed in range(1,10)]

    pola_new_agent_loaders = [
        POLAAgentLoader(path=os.path.join(current_dir, 'checkpoints', 'pola_20230921', pth), name=f'pola_new_{i}')
        for i, pth in enumerate([
            'agents_t20230920-2250_seed1_update250.pkl',
            'agents_t20230921-0058_seed2_update250.pkl',
            'agents_t20230920-2350_seed3_update250.pkl',
            'agents_t20230921-0002_seed4_update250.pkl',
            'agents_t20230920-2359_seed5_update250.pkl',
            'agents_t20230921-0152_seed6_update250.pkl',
            'agents_t20230920-2355_seed7_update250.pkl',
            'agents_t20230921-0152_seed8_update250.pkl',
            'agents_t20230921-0154_seed9_update250.pkl',
            'agents_t20230921-0325_seed10_update250.pkl'
        ])
    ]

    random_agent_loader = RandomAgentLoader(name='random')

    always_defect_agent_loader = AlwaysDefectAgentLoader(hp=hp)

    always_cooperate_agent_loader = AlwaysCooperateAgentLoader(hp=hp)

    mfos_agent_loaders = [MFOSAgentLoader(path=os.path.join(current_dir, 'checkpoints', f'mfos_seeds/new_self_3x3_{seed}/1000_0.pth'),
                                        name=f'mfos_{seed}',
                                        supported_players=[0, 1])
                         for seed in [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
                         ]
    all_agent_loaders = [*loqa_agent_loaders,
                         *pola_new_agent_loaders,
                         *loqa_rb_ablation_loader,
                         *loqa_sp_ablation_loader,
                         *advantage_alignment_loaders,
                         *advantage_alignment_mask_cooperative_empathetic,
                         *advantage_alignment_mask_empathetic,
                         *advantage_alignment_mask_spiteful,
                         *advantage_alignment_mask_vengeful,
                         *advantage_alignment_mask_vengeful_spiteful,
                         *advantage_alignment_new_baseline,
                         *mfos_agent_loaders,
                         always_defect_agent_loader, always_cooperate_agent_loader,
                         random_agent_loader,
                         ]

    return all_agent_loaders, {loader.name: loader for loader in all_agent_loaders}

def get_combinations(league_mode, all_agent_loaders, named_all_agent_loaders, hp):
    combinations = []
    assert league_mode is not None
    if league_mode == 'all':
        for i in range(len(all_agent_loaders)):
            for j in range(i, len(all_agent_loaders)):
                combinations.append((all_agent_loaders[i], all_agent_loaders[j]))
    elif league_mode == 'self_play':
        for i in range(len(all_agent_loaders)):
            agent_loader = all_agent_loaders[i]
            combinations.append((agent_loader, agent_loader))
    elif league_mode == 'trained_opponent':
        from league.trained_oponnent_loader import TrainedAgentLoader
        trained_agent_loader = TrainedAgentLoader(hp=hp, name='trained_agent')
        for i in range(len(all_agent_loaders)):
            agent_loader = all_agent_loaders[i]
            # do not add if mcts is in name
            if 'mcts' in agent_loader.name.lower():
                continue
            combinations.append((agent_loader, trained_agent_loader))
    elif league_mode == 'random_opponent':
        for i in range(len(all_agent_loaders)):
            agent_loader = all_agent_loaders[i]
            combinations.append((agent_loader, RandomAgentLoader(name='random')))
    elif league_mode == 'ablation':
        for i in range(len(all_agent_loaders)):
            for j in range(i, len(all_agent_loaders)):
                if 'ablation' in all_agent_loaders[i].name.lower():
                    combinations.append((all_agent_loaders[i], all_agent_loaders[j]))
    elif league_mode == 'ablation_2':
        for i in range(len(all_agent_loaders)):
            for j in range(i, len(all_agent_loaders)):
                if 'ablation' in all_agent_loaders[j].name.lower():
                    combinations.append((all_agent_loaders[i], all_agent_loaders[j]))
    elif league_mode == 'mfos':
        for i in range(len(all_agent_loaders)):
            for j in range(i, len(all_agent_loaders)):
                if 'mfos' in all_agent_loaders[i].name.lower() or 'mfos' in all_agent_loaders[j].name.lower():
                    combinations.append((all_agent_loaders[i], all_agent_loaders[j]))
    elif league_mode == 'advantage_alignment':
        for i in range(len(all_agent_loaders)):
            for j in range(i, len(all_agent_loaders)):
                if 'advantage_alignment' in all_agent_loaders[i].name.lower() or 'advantage_alignment' in all_agent_loaders[j].name.lower():
                    combinations.append((all_agent_loaders[i], all_agent_loaders[j]))
    elif league_mode == "advantage_alignment_mask":
        for i in range(len(all_agent_loaders)):
            for j in range(i, len(all_agent_loaders)):
                if 'advantage_alignment_mask' in all_agent_loaders[i].name.lower() or 'advantage_alignment_mask' in all_agent_loaders[j].name.lower():
                    combinations.append((all_agent_loaders[i], all_agent_loaders[j]))
    elif league_mode == 'advantage_alignment_new_baseline':
        for i in range(len(all_agent_loaders)):
            for j in range(i, len(all_agent_loaders)):
                if 'advantage_alignment_new_baseline' in all_agent_loaders[i].name.lower() or 'advantage_alignment_new_baseline' in all_agent_loaders[j].name.lower():
                    combinations.append((all_agent_loaders[i], all_agent_loaders[j]))
    elif league_mode == 'custom':
        combinations.append((named_all_agent_loaders['mfos'], named_all_agent_loaders['mfos']))
    else:
        raise NotImplementedError(f"league_mode {league_mode} not implemented")
    return combinations

def get_hp(debug_mode, batch_size, trace_length):
    hp = {
        'batch_size': batch_size,
        'height': 3,
        'width': 3,
        'g_num_actions': 4,
        'trace_length': trace_length,
        'new_coin_every_turn': False,
        'dtype': jp.float32,
        'reward_discount': 0.99,
        'lr_schedule': 'constant',
        'advantage_estimation_mode': 'gae',
        'gae_lambda': 1.0,
        'agent_entropy_beta': 5.0,
        'max_grad_norm': 10.0,
        'value_algorithm_mode': 'td0',
        'value_loss_mode': 'huber',
        'final_state_value_zero': 1,
    }

    if debug_mode:
        hp['batch_size'] = 2
        hp['trace_length'] = 5

    return hp

def main(league_name: str,
         batch_size: int = 32,
         debug_mode: bool = False,
         league_mode: str = None,
         trace_length: int = 50,
         agent1: str = None,
         agent2: str = None,):

    assert (league_mode is None and agent1 is not None and agent2 is not None) or (league_mode is not None and agent1 is None and agent2 is None), "either specify league_mode or agent1 and agent2"

    hp = get_hp(debug_mode, batch_size, trace_length)
    hp = flax.core.FrozenDict(hp)
    all_agent_loaders, named_all_agent_loaders = get_all_agent_loaders(hp=hp)
    # check all names are unique
    assert len(all_agent_loaders) == len(set([loader.name for loader in all_agent_loaders]))

    # evaluate these combinations
    if league_mode is not None:
        combinations = get_combinations(league_mode, all_agent_loaders, named_all_agent_loaders, hp)
    else:
        combinations = [(named_all_agent_loaders[agent1], named_all_agent_loaders[agent2])]
        league_mode = f'{agent1}_vs_{agent2}'

    rng = rax.PRNGKey(2)

    # print combinations
    for agent1_loader, agent2_loader in combinations:
        print(f'{agent1_loader.name} vs. {agent2_loader.name}')

    # run the evaluation of combinations
    results_matrix, episode_logs = evaluate_these_agent_combinations(combinations, hp['batch_size'], rng, hp)

    headers = [type(agent).__name__ for agent in all_agent_loaders]
    for head1 in headers:
        for head2 in headers:
            if (head1, head2) not in results_matrix:
                continue
            else:
                print(f"{head1} vs {head2}:")
                print(results_matrix[(head1, head2)])
    print(results_matrix)

    # pickle the results
    import pickle
    save_root = Path('.') / league_name
    # create the root folder to store league results if it does not exist
    save_root.mkdir(parents=True, exist_ok=True)

    with open(save_root / f'{league_mode}_results_matrix.pkl', 'wb') as f:
        pickle.dump(npify(results_matrix), f)
    with open(save_root / f'{league_mode}_episode_logs.pkl', 'wb') as f:
        pickle.dump(npify(episode_logs), f)

def test_trained_agent_loader(batch_size: int = 32, agent_name: str = 'Always Cooperate'):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    hp = {
        'batch_size': batch_size,
        'height': 3,
        'width': 3,
        'g_num_actions': 4,
        'trace_length': 50,
        'new_coin_every_turn': True,
        'dtype': jp.float32,
        'reward_discount': 0.99,
        'lr_schedule': 'constant',
        'advantage_estimation_mode': 'gae',
        'gae_lambda': 1.0,
        'agent_entropy_beta': 5.0,
        'max_grad_norm': 10.0,
        'value_algorithm_mode': 'td0',
        'value_loss_mode': 'huber',
        'final_state_value_zero': 1,
    }
    hp = flax.core.FrozenDict(hp)

    all_agent_loaders, _ = get_all_agent_loaders(hp=hp)
    # construct a dict from loader name to loader
    loader_dict = {loader.name: loader for loader in all_agent_loaders}
    combinations = [(loader_dict[agent_name], loader_dict['trained_agent'])]
    rng = rax.PRNGKey(2)
    results_matrix, episode_logs = evaluate_these_agent_combinations(combinations, batch_size=batch_size, rng=rng, hp=hp)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    fire.Fire()
