export CUDA_VISIBLE_DEVICES=1

echo "Waiting for debugger to attach..."
python3 -m debugpy --listen 10.66.12.19:1332 ./src/baselines/HIB/train.py \
    --dataset MNIST \
    --name MNIST \
    --batch_size 128 \
    --K 2 \
    --embedding_size 128 \
    --num_epoch 4 \
    --save_freq 2 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --kl_scale 1e-4 \
    --to_visualize \
    --lr 1e-4