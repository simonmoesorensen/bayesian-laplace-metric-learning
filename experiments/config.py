from typing import List


class Config:
    latent_dims: List[int]
    dataset: str
    models: List[str] = ["PFE", "DUL", "HIB", "MCDropout"]
    seeds: List[int] = [42, 43, 44, 45, 46]


class FashionMNISTConfig(Config):
    latent_dims = [2, 16, 32]
    dataset = "FashionMNIST"
    num_epoch = 150


class CIFAR10Config(Config):
    latent_dims = [16, 32, 64]
    dataset = "CIFAR10"
    num_epoch = 500


template_text = """
#!/bin/sh
### General options

### -- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J {job_name}

### -- ask for number of cores (default: 1) --
#BSUB -n 8

### -- Select the resources: 2 gpus -- 
#BSUB -gpu "num=2"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# Request GPU resources
#BSUB -R "rusage[mem={gpu_mem}GB]"
#BSUB -R "select[gpu{gpu_mem}gb]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
### BSUB -u moe.simon@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
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

export CUDA_VISIBLE_DEVICES=0,1

python3 -m src.baselines.{model}.train --dataset {dataset} --name {name} --batch_size {batch_size} --embedding_size {latent_dim} --num_epoch {num_epoch} --save_freq 100000 --gpu_id 0 1 --num_workers 8 --to_visualize --shuffle {additional_args}
"""
