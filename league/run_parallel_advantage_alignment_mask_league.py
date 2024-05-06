'''
This madness is necessary because we need to run mfos fast but on cpu :(, so we submit a bunch of jobs to slurm,
it is ugly I know. But we need to reach the rebuttal deadline.
'''
import os
import pickle
import random
import time

from run_league import *

def gen_script(run_command):
    script = "#!/bin/bash\n"
    script += "source /home/mila/a/aghajohm/repos/qadetective/env/bin/activate\n"
    script += run_command
    return script


def run_job(command, fake_submit: bool = True, partition: str = 'long-cpu'):
    script = gen_script(command)
    print('script:\n', script)
    # write this to 'sweep_temp_job.sh'
    with open('sweep_temp_job.sh', 'w') as f:
        f.write(script)

    # submit this job using slurm
    if fake_submit:
        print('fake submit')
    else:
        command = f'sbatch --time=00:10:00 --partition={partition} --mem=20Gb -c 1 sweep_temp_job.sh'
        os.system(command)


league_name = 'league_advantage_alignment_mask_2024_04_23'

def main( chunk_size: int, chunk_index: int, fake_submit: bool = True):
    command = "python run_league.py main --league_name={league_name} --debug_mode=False --agent1=\"{agent1}\" --agent2=\"{agent2}\" --trace_length=16"
    name_pairs = get_name_pairs()
    if chunk_index*chunk_size >= len(name_pairs):
        raise ValueError(f'chunk_index*chunk_size >= len(name_pairs), chunk_index: {chunk_index}, chunk_size: {chunk_size}, len(name_pairs): {len(name_pairs)}')
    name_pairs = name_pairs[chunk_index*chunk_size:(chunk_index+1)*chunk_size]
    for name_pair in name_pairs:
        agent1, agent2 = name_pair
        run_job(command=command.format(agent1=agent1, agent2=agent2, league_name=league_name), fake_submit=fake_submit, partition='long-cpu')

def get_name_pairs():
    hp = get_hp(False, 32, 16)
    hp = flax.core.FrozenDict(hp)
    all_agent_loaders, named_all_agent_loaders = get_all_agent_loaders(hp=hp)
    assert len(all_agent_loaders) == len(set([loader.name for loader in all_agent_loaders]))
    combinations = get_combinations('advantage_alignment_mask', all_agent_loaders, named_all_agent_loaders, hp)
    name_pairs = []
    for combination in combinations:
        name_pairs.append((combination[0].name, combination[1].name))
    print(f'len name_pairs: {len(name_pairs)}')
    return name_pairs

def merge_results():
    results_matrix = {}
    name_pairs = get_name_pairs()
    for name_pair in name_pairs:
        agent1, agent2 = name_pair
        file_path = Path('.') / league_name / f'{agent1}_vs_{agent2}_results_matrix.pkl'
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
        results_matrix.update(result)
    save_path = Path('.') / league_name / 'results_matrix.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results_matrix, f)


if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()









