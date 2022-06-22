#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J DUL-face-recognition

### -- ask for number of cores (default: 1) --
#BSUB -n 2

### -- Select the resources: 2 gpus -- 
#BSUB -gpu "num=2"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem=32GB]"
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

#BSUB -o logs/DUL-face-recognition.out
#BSUB -e logs/DUL-face-recognition.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.8.11
module load cuda/11.3

# Go to directory
cd /zhome/e2/5/127625/bayesian-laplace-metric-learning/src/baselines/face_recognition_pytorch

# Load venv
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

model_save_folder='./checkpoints/exp_webface_dul/'
log_tensorboard='./logtensorboard/exp_webface_dul/'

# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python3 ./train_dul.py \
    --model_save_folder $model_save_folder \
    --log_tensorboard $log_tensorboard \
    --batch_size 512 \
    --gpu_id 0 1 \
    --multi_gpu True \
    --stages 10 18 \
    --kl_scale 0.01 
