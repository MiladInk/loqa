import os
import random


def gen_script(run_command):
    script = "#!/bin/bash\n"
    script += "source ./load.sh\n"
    script += run_command
    return script


def run_job(command, fake_submit: bool = True, partition: str = 'long'):
    script = gen_script(command)
    print('script:\n', script)
    # write this to 'sweep_temp_job.sh'
    with open('sweep_temp_job.sh', 'w') as f:
        f.write(script)

    # submit this job using slurm
    if fake_submit:
        print('fake submit')
    else:
        command = f'sbatch --time=0:40:0 --partition={partition} --gres=gpu:a100l:1 --mem=20Gb -c 4 sweep_temp_job.sh'
        os.system(command)

def main(seeds, fake_submit: bool = True, partition='long'):
    commands = [
        f"python ipd.py wandb.state=enabled wandb.tags=[ipd_10_seeds_v3] hp=logits hp.eval_every=100 hp.batch_size=2048 hp.game_length=50 hp.qvalue.mode=mean hp.qvalue.train.lr_loss_qvalue=1e-2 hp.actor.train.lr_loss_actor=1e-3 hp.actor.train.entropy_beta=0.0 hp.seed={seed} hp.agent_replay_buffer.mode=disabled hp.epsilon_greedy=0.2"
        for seed in seeds
    ]
    for command in commands:
        run_job(command=command, fake_submit=fake_submit, partition=partition)


if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()









