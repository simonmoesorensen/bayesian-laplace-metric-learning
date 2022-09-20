
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=1

#node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m debugpy --listen $node_ip:1332 ./src/baselines/Laplace_online/train.py \
    --dataset FashionMNIST \
    --name train_script_full \
    --batch_size 128 \
    --embedding_size 32 \
    --num_epoch 100 \
    --save_freq 1 \
    --gpu_id 0\
    --hessian full \
    --num_workers 8 \
    --shuffle \
    --to_visualize \
    --random_seed 42