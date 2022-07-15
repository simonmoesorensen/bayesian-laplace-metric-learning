export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
python3 -m debugpy --listen 10.66.12.19:1332 ./src/baselines/HIB/train.py \
    --dataset Casia \
    --name Casia \
    --batch_size 256 \
    --K 5 \
    --embedding_size 128 \
    --num_epoch 10 \
    --save_freq 1 \
    --gpu_id 0 1\
    --num_workers 8 \
    --shuffle \
    --kl_scale 0.0001 \
    --to_visualize