export CUDA_VISIBLE_DEVICES=0

echo "Waiting for debugger to attach..."
python -m debugpy --listen 10.66.20.1:1144 src/laplace/PostHoc/train.py \
    --dataset CIFAR10 \
    --name CIFAR10 \
    --neg_margin 0.2 \
    --batch_size 16 \
    --embedding_size 128 \
    --num_epoch 20 \
    --disp_freq 2 \
    --gpu_id 0 \
    --num_workers 4 \
    --shuffle \
    --inference_model linear \
    --model_path outputs/PostHoc/checkpoints/CIFAR10/Model_Epoch_28_Time_2022-08-15T003846_checkpoint.pth \
    --to_visualize
