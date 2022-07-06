export CUDA_VISIBLE_DEVICES=0,1

model_save_folder='./checkpoints/exp_dul_dist/'
log_tensorboard='./logtensorboard/exp_dul_dist/'

echo "Waiting for debugger to attach..."
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python3 -m debugpy --listen 10.66.12.19:1334 ./train_dul_dist.py \
    --model_save_folder $model_save_folder \
    --log_tensorboard $log_tensorboard \
    --gpu_id 0 1 \
    --multi_gpu True \
    --kl_scale 0.01 \
    --batch_size 128 \
    --num_workers 8 \
    --num_epoch 60 \
    --resume_backbone checkpoints/exp_dul_dist/Backbone_IR_SE_64_DUL_Epoch_20_Batch_227440_Time_2022-06-30-09-22_checkpoint.pth \
    --resume_head checkpoints/exp_dul_dist/Head_ArcFace_Epoch_20_Batch_227440_Time_2022-06-30-09-22_checkpoint.pth \
    --resume_epoch 20
