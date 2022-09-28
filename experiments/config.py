from typing import List


class Config:
    latent_dims: List[int]
    dataset: str
    models: List[str] = ["DUL"]#, "Laplace_online"] #["Laplace_online"] #, "Backbone"] #["PFE", "Online"] #"DUL", "HIB", "MCDropout"]
    seeds: List[int] = [47]#[43, 43, 44, 45, 46]
    gpu_queue: str = "gpuv100"
    gpu_mem: str


class FashionMNISTConfig(Config):
    latent_dims = [16, 32]
    dataset = "FashionMNIST"
    num_epoch = 400
    gpu_mem = "4"


class CIFAR10Config(Config):
    latent_dims = [16, 32, 64]
    dataset = "CIFAR10"
    num_epoch = 500
    gpu_mem = "3"

class CUB200Config(Config):
    latent_dims = [64, 128]
    dataset = "CUB200"
    num_epoch = 500
    gpu_mem = "8"


class LaplaceConfig(Config):
    hessians = ["fixed"]  # , "positives", "full"]
    gpu_mem = "8"
    models = ["PostHoc"]  # , "Online"]
    gpu_queue = "gpua100"


class FashionMNISTConfigLaplace(LaplaceConfig, FashionMNISTConfig):
    num_epoch = 250


class CIFAR10ConfigLaplace(LaplaceConfig, CIFAR10Config):
    num_epoch = 250


template_text = """
#!/bin/sh
### General options

### -- specify queue --
#BSUB -q {gpu_queue}

### -- set the job Name --
#BSUB -J {job_name}

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 2 gpus -- 
#BSUB -gpu "num=1"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem={gpu_mem}GB]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
### BSUB -u moe.simon@gmail.com
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
###BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -oo {logs_dir}/run.out
#BSUB -eo {logs_dir}/run.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.8.11
module load cuda/11.7

# Load venv
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python3 -m src.baselines.{model}.train --dataset {dataset} --name {name} --batch_size {batch_size} --embedding_size {latent_dim} --num_epoch {num_epoch} --save_freq 100000 --val_freq 20 --gpu_id 0 --num_workers 8 --to_visualize --shuffle {additional_args} --linear
"""


template_text_laplace = """
#!/bin/sh
### General options

### -- specify queue --
#BSUB -q {gpu_queue}

### -- set the job Name --
#BSUB -J {job_name}

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 2 gpus -- 
#BSUB -gpu "num=1"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem={gpu_mem}GB]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
### BSUB -u moe.simon@gmail.com
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
###BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -oo {logs_dir}/run.out
#BSUB -eo {logs_dir}/run.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.8.11
module load cuda/11.7

# Load venv
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0


python3 -m src.laplace.{train_script} --dataset {dataset} --name {name} --batch_size {batch_size} --to_visualize --num_epoch {num_epoch} --embedding_size {latent_dim} --hessian {hessian} {additional_args} --linear
"""
