#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J PostHoc-MNIST

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 1 gpus -- 
#BSUB -gpu "num=1"

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

python3 -m src.laplace.train \
    --dataset MNIST \
    --name MNIST \
    --model_path outputs/Backbone/checkpoints/MNIST/MNIST/Final_Model_Epoch_100_Time_2022-08-20T100252_checkpoint.pth \
    --hessian full \
    --embedding_size 32
