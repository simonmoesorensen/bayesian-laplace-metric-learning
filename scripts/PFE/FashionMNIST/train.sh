
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=2

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m debugpy --listen $node_ip:1329 ./src/baselines/PFE/train.py \
    --dataset FashionMNIST \
    --name latent_dim_3_seed_43 \
    --batch_size 128 \
    --embedding_size 3 \
    --num_epoch 200 \
    --save_freq 20 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize \
    --linear \
    --random_seed 43