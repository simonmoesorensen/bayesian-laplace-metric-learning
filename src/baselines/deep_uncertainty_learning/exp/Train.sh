export CUDA_VISIBLE_DEVICES=0,3

model_save_folder='./checkpoints/exp_webface_dul/'
log_tensorboard='./logtensorboard/exp_webface_dul/'

echo "Waiting for debugger to attach..."
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python -m debugpy --listen 10.66.20.9:1332 ./train_dul.py \
    --model_save_folder $model_save_folder \
    --log_tensorboard $log_tensorboard \
    --gpu_id 0 1 \
    --stages 10 18 \
    --multi_gpu True \
    --kl_scale 0.01 \
    --batch_size 512 \
    --num_workers 8 \
    # --num_epoch 60 \
    # --resume_backbone checkpoints/exp_dul_dist/Backbone_IR_SE_64_DUL_Epoch_20_Batch_227440_Time_2022-06-30-09-22_checkpoint.pth \
    # --resume_head checkpoints/exp_dul_dist/Head_ArcFace_Epoch_20_Batch_227440_Time_2022-06-30-09-22_checkpoint.pth \
    # --resume_epoch 20
        
