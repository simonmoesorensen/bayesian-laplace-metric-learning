#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J laplace-online-FashionMNIST

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 2 gpus -- 
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem=12GB]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
###BSUB -u moe.simon@gmail.com
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
###BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --


#BSUB -oo outputs/Laplace_online/logs/FashionMNIST/FashionMNIST/run.out
#BSUB -eo outputs/Laplace_online/logs/FashionMNIST/FashionMNIST/run.err

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
    --name FashionMNIST_nneg_1 \
    --batch_size 128 \
    --embedding_size 32 \
    --num_epoch 1000 \
    --save_freq 20 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize \
    --random_seed 42