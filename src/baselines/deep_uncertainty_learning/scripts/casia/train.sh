export CUDA_VISIBLE_DEVICES=0,1

model_save_folder='./checkpoints/'
logs='./logtensorboard/'

echo "Waiting for debugger to attach..."
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python -m debugpy --listen 10.66.20.1:1332 ./src/train.py \
    --model_save_folder $model_save_folder \
    --log_dir $logs \
    --dataset Casia \
    --name Casia \
    --batch_size 512 \
    --embedding_size 512 \
    --arcface_scale 64 \
    --arcface_margin 28.6 \
    --num_epoch 50 \
    --save_freq 5 \
    --gpu_id 0 1\
    --num_workers 12\
    --shuffle
