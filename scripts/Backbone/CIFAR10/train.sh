export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
python3 -m debugpy --listen 10.66.12.19:1331 ./src/baselines/Backbone/train.py \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --batch_size 512 \
    --embedding_size 16 \
    --num_epoch 50 \
    --save_freq 10 \
    --gpu_id 0 1\
    --num_workers 8 \
    --shuffle \
    --to_visualize