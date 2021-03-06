export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
python3 -m debugpy --listen 10.66.20.9:1332 ./src/baselines/PFE/train.py \
    --dataset Casia \
    --name Casia \
    --batch_size 256 \
    --embedding_size 128 \
    --num_epoch 10 \
    --save_freq 1 \
    --gpu_id 0 1\
    --num_workers 8 \
    --shuffle \
    --to_visualize