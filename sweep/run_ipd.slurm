#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=cg_ios

# Remove one # to uncommment
#SBATCH --output=/network/scratch/j/juan.duque/slurm_output/slurm-%j.out
#SBATCH --error=/network/scratch/j/juan.duque/slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=0-00:29:00   
#SBATCH --gres=gpu:a100l:1

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=NONE
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=juan.duque@mila.quebec


# Submit jobs.
version=4
export WANDB_ENTITY="jduque"

module purge
eval "$(conda shell.bash hook)"
conda activate tf211-jax044-py310
module load cuda/11.1/cudnn/8.1

python ipd.py \
    hp=gru \
    hp.seed=${1} \
    wandb.state=enabled \
    hp.actor.train.lr_loss_actor=${2} \
    hp.actor.train.advantage=TD0 \
    hp.batch_size=512 \
    hp.qvalue.train.lr_loss_qvalue=${3} \
    hp.actor.train.entropy_beta=${4} \
    hp.optimizer=adam \
    hp.reset.mode=enabled \
    hp.just_self_play=True \
    hp.qvalue.mode=mean \
    hp.agent_replay_buffer.capacity=${5} \
    hp.agent_replay_buffer.update_freq=${6} \
    hp.agent_replay_buffer.mode=enabled \