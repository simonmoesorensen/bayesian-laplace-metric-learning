import argparse
from pathlib import Path

output_dir = Path(__file__).parent.parent.parent.parent / 'outputs'
baseline_dir = output_dir / 'HIE'

data_dir = Path('/work3/s174420/datasets')
vis_dir = baseline_dir / 'figures'
save_dir = baseline_dir / 'checkpoints'
log_dir = baseline_dir / 'logs'

def parse_args():
    parser = argparse.ArgumentParser(description='DUL: Data Uncertainty Learning')
    parser.add_argument('--name', type=str, default='MNIST')

    # ----- random seed for reproducing
    parser.add_argument('--random_seed', type=int, default=6666)

    # ----- directory (train & test)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--vis_dir', type=str, default=vis_dir)
    parser.add_argument('--model_save_folder', type=str, default=save_dir)
    parser.add_argument('--log_dir', type=str, default=log_dir)
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
    parser.add_argument('--to_visualize', default=False, action='store_true')
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--disp_freq', type=int, default=20)
    
    # ----- hyperparameters
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epoch', type=int, default=22)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--kl_scale', type=float, default=0.01)

    args = parser.parse_args()

    return args