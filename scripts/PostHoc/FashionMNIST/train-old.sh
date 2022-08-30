module load python3/3.8.11; module load cuda/11.7; source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,3

node_ip="$(ifconfig | grep "inet" | awk 'NR==1{print $2}')"

python3 -m debugpy --listen $node_ip:1332 ./src/laplace/post_hoc.py \
    --dataset FashionMNIST \
    --name train_script \
    --model_path results/PostHoc/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T164329_checkpoint.pth \
    --hessian fixed \
    --embedding_size 32 \
    --gpu_id 0 \
    --random_seed 46 \
    --batch_size 32 \
    --to_visualize
