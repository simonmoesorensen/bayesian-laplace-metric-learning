#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J PFE-FashionMNIST

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 2 gpus -- 
#BSUB -gpu "num=2"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"

# -- end of LSF options --

# Load the cuda module
module load python3/3.8.11
module load cuda/11.7
# Load venv
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
python3 -m src.baselines.PFE.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --batch_size 64 \
    --embedding_size 6 \
    --num_epoch 50 \
    --save_freq 10 \
    --gpu_id 0 1\
    --num_workers 8 \
    --shuffle \
    --to_visualize \
    --loss contrastive