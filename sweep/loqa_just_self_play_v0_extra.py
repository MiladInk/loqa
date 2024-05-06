import os
import random


def gen_script(run_command):
    script = "#!/bin/bash\n"
    script += "source ./load.sh\n"
    script += run_command
    return script


def run_job(command, fake_submit: bool = True):
    script = gen_script(command)
    print('script:\n', script)
    # write this to 'sweep_temp_job.sh'
    with open('sweep_temp_job.sh', 'w') as f:
        f.write(script)

    # submit this job using slurm
    if fake_submit:
        print('fake submit')
    else:
        command = 'sbatch --time=2:00:0 --partition=long --gres=gpu:a100l:1 --mem=20Gb -c 4 sweep_temp_job.sh'
        os.system(command)

def main(fake_submit: bool = True):
    commands = [
        "python coin_train.py hp=reproduce_juan wandb.state=enabled wandb.tags=[loqa_just_self_play_v0,true-waterfall-188-hp] wandb.wandb_dir=/home/mila/a/aghajohm/scratch/loqa/wandb hp.batch_size=8192 hp.save_every=1000 hp.save_dir=/home/mila/a/aghajohm/scratch/loqa hp.actor.train.lr_loss_actor=0.003 hp.qvalue.train.lr_loss_qvalue=0.01 hp.actor.train.entropy_beta=0.05 hp.actor.layers_before_gru=1 hp.actor.hidden_size=128 hp.agent_replay_buffer.mode=disabled hp.agent_replay_buffer.capacity=1 hp.agent_replay_buffer.update_freq=1 hp.just_self_play=True",
        "python coin_train.py hp=reproduce_juan wandb.state=enabled wandb.tags=[loqa_just_self_play_v0,iconic-planet-190-hp] wandb.wandb_dir=/home/mila/a/aghajohm/scratch/loqa/wandb hp.batch_size=8192 hp.save_every=1000 hp.save_dir=/home/mila/a/aghajohm/scratch/loqa hp.actor.train.lr_loss_actor=0.001 hp.qvalue.train.lr_loss_qvalue=0.01 hp.actor.train.entropy_beta=0.1 hp.actor.layers_before_gru=2 hp.actor.hidden_size=128 hp.agent_replay_buffer.mode=disabled hp.agent_replay_buffer.capacity=1 hp.agent_replay_buffer.update_freq=1 hp.just_self_play=True",
        "python coin_train.py hp=reproduce_juan wandb.state=enabled wandb.tags=[loqa_just_self_play_v0,kind-dragon-195-hp] wandb.wandb_dir=/home/mila/a/aghajohm/scratch/loqa/wandb hp.batch_size=8192 hp.save_every=1000 hp.save_dir=/home/mila/a/aghajohm/scratch/loqa hp.actor.train.lr_loss_actor=0.0003 hp.qvalue.train.lr_loss_qvalue=0.003 hp.actor.train.entropy_beta=0.3 hp.actor.layers_before_gru=2 hp.actor.hidden_size=32 hp.agent_replay_buffer.mode=disabled hp.agent_replay_buffer.capacity=1 hp.agent_replay_buffer.update_freq=1 hp.just_self_play=True",
    ]
    for command in commands:
        run_job(command= command, fake_submit=fake_submit)


if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()









