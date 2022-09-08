from config import (
    FashionMNISTConfigLaplace,
    CIFAR10ConfigLaplace,
    template_text_laplace,
)
import subprocess
from pathlib import Path

root = Path(__file__).parent.parent

for config in [FashionMNISTConfigLaplace, CIFAR10ConfigLaplace]:
    for latent_dim in config.latent_dims:
        for model in config.models:
            if model == "PostHoc":
                train_script = "train_post_hoc"
            elif model == "Online":
                train_script = "train_online"
            else:
                raise ValueError(f"Unknown model {model}")

            for hessian in config.hessians:
                for seed in config.seeds:
                    additional_args_seed = f"--random_seed {seed} "

                    if model == "PostHoc":
                        backbone_path = f"src/baselines/PFE/pretrained/{config.dataset}/latentdim_{latent_dim}_seed_{seed}.pth"
                        additional_args_backbone = f"--backbone_path {backbone_path} "
                    else:
                        additional_args_backbone = ""

                    batch_size = 16

                    name = f"{hessian}_{latent_dim}_seed_{seed}"
                    log_dir = root / "outputs" / model / "logs" / config.dataset / name
                    log_dir.mkdir(parents=True, exist_ok=True)

                    submit_script = template_text_laplace.format(
                        **{
                            "job_name": f"{model}-{config.dataset}-{name}",
                            "logs_dir": log_dir,
                            "model": model,
                            "dataset": config.dataset,
                            "name": name,
                            "batch_size": batch_size,
                            "latent_dim": latent_dim,
                            "num_epoch": config.num_epoch,
                            "gpu_queue": config.gpu_queue,
                            "gpu_mem": config.gpu_mem,
                            "hessian": hessian,
                            "additional_args": (
                                additional_args_seed + additional_args_backbone
                            ),
                            "train_script": train_script,
                        }
                    )

                    submit_file = log_dir / "script.sh"

                    with open(submit_file, "w") as f:
                        f.write(submit_script)

                    print("Submitting job:", submit_file)
                    # Execute code in terminal
                    normal = subprocess.run(
                        f"bsub < {submit_file}",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                        shell=True,
                    )
