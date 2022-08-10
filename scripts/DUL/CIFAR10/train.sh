export CUDA_VISIBLE_DEVICES=0,1 

echo "Waiting for debugger to attach..."
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python3 -m debugpy --listen 10.66.20.9:1332 ./src/baselines/DUL/train.py \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --batch_size 512 \
    --embedding_size 64 \
    --num_epoch 50 \
    --save_freq 10 \
    --gpu_id 0 1\
    --num_workers 8 \
    --shuffle \
    --to_visualize \
    --kl_scale 1e-4