export CUDA_VISIBLE_DEVICES=0,1,2,3

model_save_folder='./checkpoints/exp_webface_dul/'
log_tensorboard='./logtensorboard/exp_webface_dul/'

echo "Waiting for debugger to attach..."
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python -m debugpy --listen 10.66.20.9:1332 ./train_dul.py \
    --model_save_folder $model_save_folder \
    --log_tensorboard $log_tensorboard \
    --gpu_id 0 1 2 3 \
    --multi_gpu True \
    --stages 10 18 \
    --kl_scale 0.01 \
    --batch_size 512
