export CUDA_VISIBLE_DEVICES=0,3

model_save_folder='./checkpoints/mnist/'
log_tensorboard='./logtensorboard/mnist/'

echo "Waiting for debugger to attach..."
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python -m debugpy --listen 10.66.20.9:1332 ./train_dul.py \
    --model_save_folder $model_save_folder \
    --log_tensorboard $log_tensorboard \
    --batch_size 128 \
    --dataset MNIST \
    --input_size 28,28 \
    --num_epoch 5 \
    --save_freq 5 \
    --gpu_id 0 1 \
    --multi_gpu True \
    --num_workers 8
