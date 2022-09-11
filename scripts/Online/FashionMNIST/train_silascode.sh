
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

#node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m src.laplace.online \
    --dataset FashionMNIST \
    --name train_script_32_42_no_opt \
    --hessian fixed \
    --embedding_size 32 \
    --gpu_id 0 \
    --random_seed 42 \
    --batch_size 16 \
    --to_visualize
