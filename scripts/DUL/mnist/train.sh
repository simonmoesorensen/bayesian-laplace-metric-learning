export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python3 -m debugpy --listen 10.66.20.1:1332 ./src/baselines/DUL/train.py \
    --dataset MNIST \
    --name MNIST \
    --batch_size 512 \
    --embedding_size 256 \
    --arcface_scale 15 \
    --arcface_margin 0.8 \
    --num_epoch 1 \
    --save_freq 1 \
    --gpu_id 0 1\
    --num_workers 8 \
    --shuffle \
    --to_visualize