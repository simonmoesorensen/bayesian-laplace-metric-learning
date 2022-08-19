#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J PostHoc-CIFAR10

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
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T133008_checkpoint.pth \
    --hessian fixed \
    --embedding_size 16
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T133021_checkpoint.pth \
    --hessian fixed \
    --embedding_size 16
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T133045_checkpoint.pth \
    --hessian fixed \
    --embedding_size 16
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T134059_checkpoint.pth \
    --hessian fixed \
    --embedding_size 16
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T134120_checkpoint.pth \
    --hessian fixed \
    --embedding_size 16

python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T134310_checkpoint.pth \
    --hessian fixed \
    --embedding_size 32
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T135154_checkpoint.pth \
    --hessian fixed \
    --embedding_size 32
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T135210_checkpoint.pth \
    --hessian fixed \
    --embedding_size 32
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T135437_checkpoint.pth \
    --hessian fixed \
    --embedding_size 32
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T140253_checkpoint.pth \
    --hessian fixed \
    --embedding_size 32

python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T140256_checkpoint.pth \
    --hessian fixed \
    --embedding_size 64
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T140559_checkpoint.pth \
    --hessian fixed \
    --embedding_size 64
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T141355_checkpoint.pth \
    --hessian fixed \
    --embedding_size 64
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T141349_checkpoint.pth \
    --hessian fixed \
    --embedding_size 64
python3 -m src.laplace.train \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --model_path outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-19T141859_checkpoint.pth \
    --hessian fixed \
    --embedding_size 64
