
module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m debugpy --listen $node_ip:1332 ./src/laplace/train_post_hoc.py \
    --dataset FashionMNIST \
    --name train_script \
    --backbone_path src/baselines/PFE/pretrained/FashionMNIST/latentdim_2_seed_46.pth \
    --hessian fixed \
    --embedding_size 2 \
    --gpu_id 0 \
    --random_seed 46 \
    --batch_size 16 \
    --to_visualize
