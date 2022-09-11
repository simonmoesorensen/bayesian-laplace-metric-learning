#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J laplace-online-FashionMNIST

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 2 gpus -- 
#BSUB -gpu "num=1"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "select[gpu12gb]"

# -- end of LSF options --

# Load the cuda module
module load python3/3.8.11
module load cuda/11.7
# Load venv
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

echo "Waiting for debugger to attach..."
python3 -m src.baselines.Laplace_online.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --batch_size 128 \
    --embedding_size 32 \
    --num_epoch 100 \
    --save_freq 20 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize \