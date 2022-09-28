
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m debugpy --listen $node_ip:1329 ./src/baselines/MCDropout/train.py \
    --dataset FashionMNIST \
    --name train_script \
    --batch_size 64 \
    --embedding_size 3 \
    --num_epoch 1 \
    --save_freq 2 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize