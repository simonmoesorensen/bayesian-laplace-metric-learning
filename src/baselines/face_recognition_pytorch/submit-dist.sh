#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J DUL-dist

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 4 gpus -- 
#BSUB -gpu "num=4"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"

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

#BSUB -o logs_dist/DUL-face-recognition-run1.out
#BSUB -e logs_dist/DUL-face-recognition-run1.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.8.11
module load cuda/11.3

# Go to directory
cd /zhome/e2/5/127625/bayesian-laplace-metric-learning/src/baselines/face_recognition_pytorch

# Load venv
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3

model_save_folder='./checkpoints/exp_dul_dist/'
log_tensorboard='./logtensorboard/exp_dul_dist/'

# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python3 ./train_dul_dist.py \
    --model_save_folder $model_save_folder \
    --log_tensorboard $log_tensorboard \
    --batch_size 512 \
    --gpu_id 0 1 2 3 \
    --multi_gpu True \
    --stages 10 18 \
    --kl_scale 0.01 \
    --lr 0.1 \
    --num_workers 8
    

# --num_epoch 60 \
# --resume_backbone checkpoints/exp_webface_dul/Backbone_IR_SE_64_DUL_Epoch_32_Batch_113720_Time_2022-06-26-03-03_checkpoint.pth \
# --resume_head checkpoints/exp_webface_dul/Head_ArcFace_Epoch_32_Batch_113720_Time_2022-06-26-03-03_checkpoint.pth \
# --resume_epoch 32
    