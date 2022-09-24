
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m debugpy --listen $node_ip:1332 ./src/baselines/Backbone/train.py \
    --dataset FashionMNIST \
    --name latent_dim_2_seed_42 \
    --batch_size 512 \
    --embedding_size 2 \
    --num_epoch 200 \
    --save_freq 20 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize \
    --random_seed 42