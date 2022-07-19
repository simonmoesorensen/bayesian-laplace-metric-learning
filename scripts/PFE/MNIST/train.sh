export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
python3 -m debugpy --listen 10.66.20.9:1333 ./src/baselines/PFE/train.py \
    --dataset MNIST \
    --name MNIST \
    --batch_size 64 \
    --embedding_size 128 \
    --num_epoch 3 \
    --save_freq 1 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize