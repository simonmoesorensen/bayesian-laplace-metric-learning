export CUDA_VISIBLE_DEVICES=0

echo "Waiting for debugger to attach..."
python3 -m debugpy --listen 10.66.20.9:1331 ./src/baselines/Backbone/train.py \
    --dataset MNIST \
    --name MNIST \
    --batch_size 512 \
    --embedding_size 4 \
    --num_epoch 20 \
    --save_freq 10 \
    --gpu_id 0\
    --num_workers 8 \
    --shuffle \
    --to_visualize