export CUDA_VISIBLE_DEVICES=3

echo "Waiting for debugger to attach..."
python3 -m debugpy --listen 10.66.20.9:1331 ./src/baselines/Backbone/train.py \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --batch_size 128 \
    --embedding_size 512 \
    --num_epoch 20 \
    --save_freq 2 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize