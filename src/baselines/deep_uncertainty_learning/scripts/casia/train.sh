export CUDA_VISIBLE_DEVICES=0,1

model_save_folder='./checkpoints/'
logs='./logtensorboard/'

echo "Waiting for debugger to attach..."
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python -m debugpy --listen 10.66.12.19:1332 ./src/train.py \
    --model_save_folder $model_save_folder \
    --log_dir $logs \
    --dataset Casia \
    --name Casia \
    --batch_size 512 \
    --embedding_size 512 \
    --arcface_scale 60 \
    --arcface_margin 0.5 \
    --num_epoch 20 \
    --save_freq 1 \
    --gpu_id 0 1\
    --num_workers 12
