export CUDA_VISIBLE_DEVICES=0,1

echo "Waiting for debugger to attach..."
python -m debugpy --listen 10.66.20.1:1143 src/laplace/PostHoc/train.py
