export CUDA_VISIBLE_DEVICES=0

echo "Waiting for debugger to attach..."
python3 -m debugpy --listen 10.66.12.19:1332 ./src/baselines/PFE/train.py \
    --dataset FashionMNIST \
    --name FashionMNIST \
    --batch_size 128 \
    --embedding_size 2 \
    --num_epoch 1 \
    --save_freq 2 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize