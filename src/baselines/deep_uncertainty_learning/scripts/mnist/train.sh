export CUDA_VISIBLE_DEVICES=0,1,2,3

model_save_folder='./checkpoints/'
logs='./logtensorboard/'

echo "Waiting for debugger to attach..."
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python -m debugpy --listen 10.66.20.9:1332 ./src/train.py \
    --model_save_folder $model_save_folder \
    --log_dir $logs \
    --dataset MNIST \
    --name MNIST \
    --batch_size 512 \
    --embedding_size 256 \
    --arcface_scale 15 \
    --arcface_margin 0.8 \
    --num_epoch 10 \
    --save_freq 5 \
    --gpu_id 0 3\
    --num_workers 8 \
    --shuffle