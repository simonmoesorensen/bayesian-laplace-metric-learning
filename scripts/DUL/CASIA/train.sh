
# notice: default kl_scale is 0.01 in DUL (base on original paper) 
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m debugpy --listen $node_ip:1332 ./src/baselines/DUL/train.py \
    --dataset Casia \
    --name train_script \
    --batch_size 512 \
    --embedding_size 512 \
    --arcface_scale 64 \
    --arcface_margin 28.6 \
    --num_epoch 1 \
    --save_freq 1 \
    --gpu_id 0 1\
    --num_workers 12\
    --shuffle\
    --to_visualize
