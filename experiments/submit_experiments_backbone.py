import subprocess
from pathlib import Path

from config import CIFAR10Config, FashionMNISTConfig, template_text

root = Path(__file__).parent.parent

model = "Backbone"
# Submit FashionMNIST experiments
for config in [FashionMNISTConfig, CIFAR10Config]:
    batch_size = 512

    if config.dataset == "FashionMNIST":
        gpu_mem = "16"
        num_epoch = 50
    elif config.dataset == "CIFAR10":
        gpu_mem = "32"
        num_epoch = 150

    for latent_dim in config.latent_dims:
        for seed in config.seeds:
            name = f"latentdim_{latent_dim}_seed_{seed}"
            log_dir = root / "outputs" / model / "logs" / config.dataset / name
            log_dir.mkdir(parents=True, exist_ok=True)

            submit_script = template_text.format(
                **{
                    "job_name": f"{model}-{config.dataset}-{name}",
                    "logs_dir": log_dir,
                    "model": model,
                    "dataset": config.dataset,
                    "name": name,
                    "batch_size": batch_size,
                    "latent_dim": latent_dim,
                    "num_epoch": num_epoch,
                    "additional_args": f"--random_seed {seed}",
                    "gpu_mem": gpu_mem,
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
