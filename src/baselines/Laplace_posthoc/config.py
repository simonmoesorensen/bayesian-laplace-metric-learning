import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

output_dir = Path(__file__).parent.parent.parent.parent / "outputs"
baseline_dir = output_dir / "Laplace_posthoc"

data_dir = Path(os.getenv("DATA_DIR"))
vis_dir = baseline_dir / "figures"
save_dir = baseline_dir / "checkpoints"
log_dir = baseline_dir / "logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Laplace posthoc")
    parser.add_argument("--name", type=str, default="MNIST")

    # ----- random seed for reproducing
    parser.add_argument("--random_seed", type=int, default=6666)

    # ----- directory (train & test)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_dir", type=str, default=data_dir)
    parser.add_argument("--vis_dir", type=str, default=vis_dir)
    parser.add_argument("--model_save_folder", type=str, default=save_dir)
    parser.add_argument("--log_dir", type=str, default=log_dir)
    parser.add_argument("--save_freq", type=int, default=4)

    # ----- training env
    parser.add_argument("--multi_gpu", type=bool, default=True)
    parser.add_argument("--gpu_id", type=str, nargs="+")

    # ----- resume pretrain details
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--model_path", type=str, default=None)

    # ----- model & training details
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--to_visualize", default=False, action="store_true")
    parser.add_argument("--disp_freq", type=int, default=20)
    parser.add_argument("--linear", default=False, action="store_true")

    # ----- hessian details
    parser.add_argument("--hessian", type=str, default="full")
    parser.add_argument("--test_samples", type=int, default=100)

    # ----- data loader details
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shuffle", default=False, action="store_true")
    parser.add_argument("--pin_memory", default=True, action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)

    # ----- hyperparameters
    parser.add_argument("--num_epoch", type=int, default=22)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    return args
