import os
import random


def gen_script(config):
    script = "#!/bin/bash\n"
    script += "source ./load.sh\n"
    script += "python coin_train.py "
    for key, value in config.items():
        script += "{}={} ".format(key, value)
    return script


def run_random_job(fake_submit: bool = True):
    hparams = {
        'hp.actor.train.lr_loss_actor': [3e-3, 1e-3, 3e-4],
        'hp.qvalue.train.lr_loss_qvalue': [1e-2, 3e-3],
        'hp.actor.train.entropy_beta': [0.01, 0.05, 0.1, 0.3],
        'hp.actor.layers_before_gru': [1, 2],
        'hp.actor.hidden_size': [32, 64, 128],
        'hp.agent_replay_buffer.mode': ['disabled', 'enabled'],
        'hp.agent_replay_buffer.capacity': [10000],
        'hp.agent_replay_buffer.update_freq': [1, 2, 10],
        'hp.just_self_play': ['True', 'False'],
    }

    # sample a random config
    config = {
        'hp': 'reproduce_juan',
        'wandb.state': 'enabled',
        'wandb.tags': ['loqa_just_self_play_v0'],
        'wandb.wandb_dir': '/home/mila/a/aghajohm/scratch/loqa/',
        'hp.batch_size': 8192,
        'hp.save_every': 1000,
        'hp.save_dir': '/home/mila/a/aghajohm/scratch/loqa',
    }
    for key, values in hparams.items():
        config[key] = random.choice(values)

    if config['hp.agent_replay_buffer.mode'] == 'disabled':
        config['hp.agent_replay_buffer.capacity'] = 1
        config['hp.agent_replay_buffer.update_freq'] = 1

    script = gen_script(config)
    print('script: ', script)
    # write this to 'sweep_temp_job.sh'
    with open('sweep_temp_job.sh', 'w') as f:
        f.write(script)

    # submit this job using slurm
    if fake_submit:
        print('fake submit')
    else:
        command = 'sbatch --time=2:00:0 --partition=long --gres=gpu:a100l:1 --mem=20Gb -c 4 sweep_temp_job.sh'
        os.system(command)

def main(num_jobs: int, fake_submit: bool = True):
    for i in range(num_jobs):
        run_random_job(fake_submit=fake_submit)

if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()









