#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J Backbone-CIFAR10

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 2 gpus -- 
#BSUB -gpu "num=2"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "select[gpu40gb]"

# -- end of LSF options --

# Load the cuda module
module load python3/3.8.11
module load cuda/11.7
# Load venv
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python3 -m src.baselines.Backbone.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --batch_size 512 \
    --embedding_size 32 \
    --num_epoch 500 \
    --save_freq 50 \
    --gpu_id 0 \
    --num_workers 8 \
    --disp_freq 2 \
    --shuffle \
    --to_visualize