#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J DUL-cifar10

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 2 gpus -- 
#BSUB -gpu "num=2"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem=40GB]"
#BSUB -R "select[gpu40gb]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u moe.simon@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o logs/cifar10/run1.out
#BSUB -e logs/cifar10/run1.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.8.11
module load cuda/11.7

# Go to directory
cd /zhome/e2/5/127625/bayesian-laplace-metric-learning/src/baselines/deep_uncertainty_learning

# Load venv
source /zhome/e2/5/127625/bayesian-laplace-metric-learning/src/baselines/deep_uncertainty_learning/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

model_save_folder='./checkpoints/'
logs='./logtensorboard/'

# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python3 ./src/train.py \
    --model_save_folder $model_save_folder \
    --log_dir $logs \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --batch_size 512 \
    --embedding_size 256 \
    --arcface_scale 15 \
    --arcface_margin 0.8 \
    --num_epoch 40 \
    --save_freq 5 \
    --gpu_id 0 1\
    --num_workers 12 \
    --shuffle \
    --to_visualize
