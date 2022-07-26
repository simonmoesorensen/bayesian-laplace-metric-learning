export CUDA_VISIBLE_DEVICES=0

echo "Waiting for debugger to attach..."
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python3 -m debugpy --listen 10.66.12.19:1331 ./src/baselines/DUL/train.py \
    --dataset MNIST \
    --name MNIST \
    --batch_size 512 \
    --embedding_size 64 \
    --arcface_scale 4 \
    --arcface_margin 10 \
    --num_epoch 30 \
    --save_freq 10 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize \
    --kl_scale 1e-3
