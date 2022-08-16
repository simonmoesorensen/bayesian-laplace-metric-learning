export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
python -m debugpy --listen 10.66.20.1:1144 src/laplace/PostHoc/train.py \
    --dataset MNIST \
    --name MNIST \
    --neg_margin 0.2 \
    --batch_size 32 \
    --embedding_size 128 \
    --num_epoch 5 \
    --disp_freq 2 \
    --gpu_id 0 \
    --num_workers 12 \
    --shuffle \
    --inference_model linear \
    --model_path outputs/PostHoc/checkpoints/MNIST/Model_Epoch_28_Time_2022-08-16T175053_checkpoint.pth \
    --to_visualize
