export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
python -m debugpy --listen 10.66.20.1:1143 src/laplace/PostHoc/train.py \
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
    --to_visualize
