
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m debugpy --listen $node_ip:1332 ./src/laplace/train_post_hoc.py \
    --dataset CIFAR10 \
    --name train_script \
    --backbone_path src/baselines/PFE/pretrained/CIFAR10/latentdim_64_seed_42.pth \
    --hessian fixed \
    --embedding_size 64 \
    --gpu_id 0 \
    --random_seed 42 \
    --batch_size 32 \
    --to_visualize
