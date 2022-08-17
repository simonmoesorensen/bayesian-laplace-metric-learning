
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print 1332}')"

python3 -m debugpy --listen $node_ip:1332 ./src/baselines/HIB/train.py \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --batch_size 256 \
    --K 3 \
    --embedding_size 512 \
    --num_epoch 20 \
    --save_freq 1 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --kl_scale 1e-4 \
    --to_visualize