module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m debugpy --listen $node_ip:1340 ./src/baselines/Backbone/train.py \
    --dataset CUB200 \
    --name train_script \
    --batch_size 64 \
    --embedding_size 128 \
    --num_epoch 500 \
    --val_freq 20 \
    --save_freq 10000 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize