export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
python3 -m debugpy --listen 10.66.20.9:1332 ./src/baselines/PFE/train.py \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --batch_size 128 \
    --embedding_size 256 \
    --num_epoch 20 \
    --save_freq 25 \
    --gpu_id 0 1\
    --num_workers 8 \
    --shuffle \
    --to_visualize