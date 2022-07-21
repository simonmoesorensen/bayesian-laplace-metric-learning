export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
python3 -m debugpy --listen 10.66.12.19:1332 ./src/baselines/HIB/train.py \
    --dataset MNIST \
    --name MNIST \
    --batch_size 64 \
    --K 8 \
    --embedding_size 128 \
    --num_epoch 6 \
    --save_freq 2 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --kl_scale 0.001 \
    --to_visualize