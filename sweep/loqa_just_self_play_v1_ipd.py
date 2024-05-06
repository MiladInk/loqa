import os
import random


def gen_script(config):
    script = "#!/bin/bash\n"
    script += "source ./load.sh\n"
    script += "python coin_train.py "
    for key, value in config.items():
        script += "{}={} ".format(key, value)
    return script

def gen_command(config):
    command = "sbatch --job-name=ipd_loqa ./sweep/run_ipd.slurm 1"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'hp.actor.train.lr_loss_actor': [3e-3, 1e-3, 3e-4],
        'hp.qvalue.train.lr_loss_qvalue': [1e-4, 3e-4, 1e-3, 3e-3],
        'hp.actor.train.entropy_beta': [0.05, 0.1, 0.3],
        'hp.agent_replay_buffer.capacity': [10000],
        'hp.agent_replay_buffer.update_freq': [10],
    }

    # sample a random config
    config = {}
    for key, values in hparams.items():
        config[key] = random.choice(values)

    # if config['hp.agent_replay_buffer.mode'] == 'disabled':
    #     config['hp.agent_replay_buffer.capacity'] = 1
    #     config['hp.agent_replay_buffer.update_freq'] = 1

    script = gen_script(config)
    print('script: ', script)
    # write this to 'sweep_temp_job.sh'
    with open('sweep_temp_job.sh', 'w') as f:
        f.write(script)

    # submit this job using slurm
    if fake_submit:
        print('fake submit')
    else:
        command = gen_command(config)
        os.system(command)

def main(num_jobs: int, fake_submit: bool = True):
    for i in range(num_jobs):
        run_random_job(fake_submit=fake_submit)

if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()









