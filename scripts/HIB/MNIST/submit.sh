#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J DUL-mnist

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

#BSUB -o logs/mnist/run1.out
#BSUB -e logs/mnist/run1.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.8.11
module load cuda/11.7

# Go to directory
cd /zhome/e2/5/127625/bayesian-laplace-metric-learning

# Load venv
source /zhome/e2/5/127625/bayesian-laplace-metric-learning/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
python3 ./src/baselines/HIB/train.py \
    --dataset MNIST \
    --name MNIST \
    --batch_size 64 \
    --K 8 \
    --embedding_size 128 \
    --num_epoch 3 \
    --save_freq 1 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --kl_scale 0.0001 \
    --to_visualize