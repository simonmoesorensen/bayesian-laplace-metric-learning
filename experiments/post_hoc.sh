#!/bin/bash
#BSUB -J post_hoc
#BSUB -o post_hoc_%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -u s174433@student.dtu.dk

module load python3/3.9.6
module load cuda/11.3
source venv/bin/activate

# python src/laplace/metric_learning.py
python src/laplace/post_hoc.py
python src/laplace/evaluate.py
python src/visualization/plot_ood.py
python src/visualization/plot_roc.py
