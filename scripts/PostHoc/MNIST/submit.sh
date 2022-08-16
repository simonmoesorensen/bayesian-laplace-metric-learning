#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J PostHoc-MNIST

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 2 gpus -- 
#BSUB -gpu "num=1"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "span[hosts=1]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s174433@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

## BSUB -oo logs/PostHoc/cifar/run1.out
## BSUB -eo logs/PostHoc/cifar/run1.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.9.6
module load cuda/11.3
source venv/bin/activate

# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0

python src/laplace/PostHoc/train.py \
    --dataset MNIST \
    --name MNIST \
    --neg_margin 0.2 \
    --batch_size 32 \
    --embedding_size 128 \
    --num_epoch 30 \
    --disp_freq 2 \
    --gpu_id 0 \
    --num_workers 12 \
    --shuffle \
    --to_visualize