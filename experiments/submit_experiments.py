import subprocess
from pathlib import Path

from config import CIFAR10Config, template_text

root = Path(__file__).parent.parent

# Submit FashionMNIST experiments
for config in [CIFAR10Config]:
    for latent_dim in config.latent_dims:
        for model in config.models:
            batch_size = 256

            if model == "DUL":
                if config.dataset == "FashionMNIST":
                    additional_args = "--kl_scale 1e-5"
                elif config.dataset == "CIFAR10":
                    additional_args = "--kl_scale 1e-7"

            elif model == "HIB":
                additional_args = "--kl_scale 1e-6 --K 8"
            else:
                additional_args = ""

            for seed in config.seeds:
                name = f"latentdim_{latent_dim}_seed_{seed}"
                log_dir = root / "outputs" / model / "logs" / config.dataset / name
                log_dir.mkdir(parents=True, exist_ok=True)
                additional_args_seed = additional_args + f" --random_seed {seed}"

                submit_script = template_text.format(
                    **{
                        "job_name": f"{model}-{config.dataset}-{name}",
                        "logs_dir": log_dir,
                        "model": model,
                        "dataset": config.dataset,
                        "name": name,
                        "batch_size": batch_size,
                        "latent_dim": latent_dim,
                        "num_epoch": config.num_epoch,
                        "additional_args": additional_args_seed,
                        "gpu_mem": config.gpu_mem,
                        "gpu_queue": config.gpu_queue,
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
