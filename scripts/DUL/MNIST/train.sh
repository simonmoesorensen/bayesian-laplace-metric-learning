
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print 1331}')"

python3 -m debugpy --listen $node_ip:1332 ./src/baselines/DUL/train.py \
    --dataset MNIST \
    --name MNIST \
    --batch_size 512 \
    --embedding_size 6 \
    --num_epoch 30 \
    --save_freq 15 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize \
    --kl_scale 1e-4
