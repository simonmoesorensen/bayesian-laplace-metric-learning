export CUDA_VISIBLE_DEVICES=1,3

model_save_folder='./checkpoints/exp_dul_dist/'
log_tensorboard='./logtensorboard/exp_dul_dist/'

echo "Waiting for debugger to attach..."
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python -m debugpy --listen 10.66.20.9:1335 ./train_dul_dist.py \
    --model_save_folder $model_save_folder \
    --log_tensorboard $log_tensorboard \
    --gpu_id 0 1 \
    --multi_gpu True \
    --kl_scale 0.01 \
    --batch_size 256 \
    --num_workers 8
