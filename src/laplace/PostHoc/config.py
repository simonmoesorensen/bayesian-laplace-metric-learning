import argparse
from pathlib import Path

baseline_dir = Path("outputs") / "PostHoc"

data_dir = Path("/work3/s174433/data")
vis_dir = baseline_dir / "figures"
save_dir = baseline_dir / "checkpoints"
log_dir = baseline_dir / "logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc Laplace approximation")
    parser.add_argument("--name", type=str, default="CIFAR10")

    # ----- random seed for reproducing
    parser.add_argument("--random_seed", type=int, default=6666)

    # ----- directory (train & test)
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--data_dir", type=str, default=data_dir)
    parser.add_argument("--vis_dir", type=str, default=vis_dir)
    parser.add_argument("--model_save_folder", type=str, default=save_dir)
    parser.add_argument("--log_dir", type=str, default=log_dir)
    parser.add_argument("--save_freq", type=int, default=4)

    # ----- training env
    parser.add_argument("--multi_gpu", type=bool, default=True)
    parser.add_argument("--gpu_id", type=str, nargs="+", default=["0"])

    # ----- resume pretrain details
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--model_path", type=str, default=None)

    # ----- model & training details
    parser.add_argument("--head_name", type=str, default="ArcFace")
    parser.add_argument("--loss_name", type=str, default="Softmax")
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--to_visualize", default=False, action="store_true")
    parser.add_argument("--disp_freq", type=int, default=20)

    # ----- data loader details
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shuffle", default=False, action="store_true")
    parser.add_argument("--pin_memory", default=True, action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)

    # ----- laplace details
    parser.add_argument("--neg_margin", type=float, default=0.2)
    parser.add_argument("--inference_model", type=str, default="linear")
    parser.add_argument("--hessian_calculator", type=str, default="")
    parser.add_argument("--posterior_samples", type=int, default=12)

    # ----- hyperparameters
    parser.add_argument("--num_epoch", type=int, default=22)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    return args
