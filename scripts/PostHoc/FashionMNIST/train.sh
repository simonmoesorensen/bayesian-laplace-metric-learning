
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

#node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m debugpy --listen $node_ip:1337 ./src/baselines/Laplace_posthoc/train.py \
    --dataset FashionMNIST \
    --name latent_dim_3_seed_42_opt \
    --model_path src/baselines/PFE/pretrained/FashionMNIST/conv/latentdim_3_seed_42.pth \
    --hessian full \
    --embedding_size 3 \
    --gpu_id 0 \
    --random_seed 42 \
    --batch_size 64 \
    --to_visualize
