#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J PostHoc-FashionMNIST

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
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T125948_checkpoint.pth \
    --hessian fixed \
    --embedding_size 2
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T125949_checkpoint.pth \
    --hessian fixed \
    --embedding_size 2
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T125948_checkpoint.pth \
    --hessian fixed \
    --embedding_size 2
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T130238_checkpoint.pth \
    --hessian fixed \
    --embedding_size 2
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T130238_checkpoint.pth \
    --hessian fixed \
    --embedding_size 2

python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T174002_checkpoint.pth \
    --hessian fixed \
    --embedding_size 16
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T174210_checkpoint.pth \
    --hessian fixed \
    --embedding_size 16
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T174418_checkpoint.pth \
    --hessian fixed \
    --embedding_size 16
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T174627_checkpoint.pth \
    --hessian fixed \
    --embedding_size 16
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T174834_checkpoint.pth \
    --hessian fixed \
    --embedding_size 16

python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T164329_checkpoint.pth \
    --hessian fixed \
    --embedding_size 32
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T164535_checkpoint.pth \
    --hessian fixed \
    --embedding_size 32
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T164740_checkpoint.pth \
    --hessian fixed \
    --embedding_size 32
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T164946_checkpoint.pth \
    --hessian fixed \
    --embedding_size 32
python3 -m src.laplace.train \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --model_path outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T165154_checkpoint.pth \
    --hessian fixed \
    --embedding_size 32
