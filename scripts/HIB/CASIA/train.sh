
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m debugpy --listen $node_ip:1332 ./src/baselines/HIB/train.py \
    --dataset Casia \
    --name train_script \
    --batch_size 256 \
    --K 5 \
    --embedding_size 128 \
    --num_epoch 10 \
    --save_freq 1 \
    --gpu_id 0 1\
    --num_workers 8 \
    --shuffle \
    --kl_scale 0.00015 \
    --to_visualize