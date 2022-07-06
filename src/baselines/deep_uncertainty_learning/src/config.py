import argparse
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.parent
data_dir = Path('/work3/s174420/datasets')
test_dir = data_dir / 'validation'

def parse_args():
    parser = argparse.ArgumentParser(description='DUL: Data Uncertainty Learning for MNIST')
    parser.add_argument('--name', type=str, default='MNIST')

    # ----- random seed for reproducing
    parser.add_argument('--random_seed', type=int, default=6666)

    # ----- directory (train & test)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--model_save_folder', type=str, default='./checkpoints/')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_freq', type=int, default=4)

    # ----- training env
    parser.add_argument('--multi_gpu', type=bool, default=True)
    parser.add_argument('--gpu_id', type=str, nargs='+')
    
    # ----- resume pretrain details
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--model_path', type=str, default=None)
    
    # ----- model & training details
    parser.add_argument('--head_name', type=str, default='ArcFace')
    parser.add_argument('--loss_name', type=str, default='Softmax')
    parser.add_argument('--triplet_margin', type=float, default=0.2)
    parser.add_argument('--arcface_scale', type=int, default=64)
    parser.add_argument('--arcface_margin', type=float, default=0.5)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--to_visualize', type=bool, default=False)
    
    # ----- hyperparameters
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epoch', type=int, default=22)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kl_scale', type=float, default=0.01)

    args = parser.parse_args()

    return args