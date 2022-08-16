export CUDA_VISIBLE_DEVICES=0

echo "Waiting for debugger to attach..."
python3 -m debugpy --listen 10.66.20.1:1333 ./src/baselines/MCDropout/train.py \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --batch_size 64 \
    --embedding_size 6 \
    --num_epoch 20 \
    --save_freq 1 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize