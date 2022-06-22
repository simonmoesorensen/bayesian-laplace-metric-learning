
export CUDA_VISIBLE_DEVICES=3

logs_test_file='./logs_test/testfr_webface_dul.log'

model_for_test=''

echo "Waiting for debugger to attach..."

python -m debugpy --listen 10.66.20.1:1332 --wait-for-client ./test_fr_dul.py 
