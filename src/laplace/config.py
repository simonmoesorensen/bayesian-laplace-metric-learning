import argparse
from pathlib import Path

baseline_dir = Path("outputs") / "PostHoc"

data_dir = Path("/work3/s174433/datasets")
vis_dir = baseline_dir / "figures"
save_dir = baseline_dir / "checkpoints"
log_dir = baseline_dir / "logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc Laplace approximation")
    parser.add_argument("--name", type=str, default="FashionMNIST")

    # ----- random seed for reproducing
    parser.add_argument("--random_seed", type=int, default=6666)

    # ----- directory (train & test)
    parser.add_argument("--dataset", type=str, default="FashionMNIST")
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
    # parser.add_argument("--model_path", type=str, default="outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T160116_checkpoint.pth")
    # parser.add_argument("--model_path", type=str, default="outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-19T160331_checkpoint.pth")
    # 64 dim CIFAR10, two layer linear
    # parser.add_argument("--model_path", type=str, default="outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-18T173105_checkpoint.pth")
    # 64 dim CIFAR10, one layer linear
    # parser.add_argument("--model_path", type=str, default="outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-18T170317_checkpoint.pth")
    # 32 dim CIFAR10
    # parser.add_argument("--model_path", type=str, default="outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-18T163031_checkpoint.pth")
    # 16 dim CIFAR10
    # parser.add_argument("--model_path", type=str, default="outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-18T163509_checkpoint.pth")
    # 2 dim CIFAR10
    # parser.add_argument("--model_path", type=str, default="outputs/Backbone/checkpoints/CIFAR10/CIFAR10/Final_Model_Epoch_500_Time_2022-08-18T164603_checkpoint.pth")
    # 32 dim FashionMNIST
    # parser.add_argument("--model_path", type=str, default="outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-18T153444_checkpoint.pth")
    # 16 dim FashionMNIST
    # parser.add_argument("--model_path", type=str, default="outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-18T154748_checkpoint.pth")
    # 2 dim FashionMNIST
    # parser.add_argument("--model_path", type=str, default="outputs/Backbone/checkpoints/FashionMNIST/FashionMNIST/Final_Model_Epoch_100_Time_2022-08-18T160018_checkpoint.pth")


    # ----- model & training details
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
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--inference_model", type=str, default="linear")
    parser.add_argument("--hessian", type=str, default="full")
    parser.add_argument("--posterior_samples", type=int, default=3)

    # ----- hyperparameters
    parser.add_argument("--num_epoch", type=int, default=250)
    parser.add_argument("--hessian_freq", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    return args
