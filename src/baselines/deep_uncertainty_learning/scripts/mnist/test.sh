export CUDA_VISIBLE_DEVICES=3

logs_test_file='./logs_test/testfr_ms1m_dul.log'

model_for_test='checkpoints/mnist/Backbone_MNIST_Epoch_5_Batch_2340_Time_2022-07-06-14-49_checkpoint.pth'

python -m debugpy --listen 10.66.20.9:1332  ./test_fr_dul.py \
    --model_for_test $model_for_test \
    --dataset MNIST