from config import FashionMNISTConfig, CIFAR10Config, template_text
import subprocess
from pathlib import Path

root = Path(__file__).parent.parent

# Submit FashionMNIST experiments
for config in [FashionMNISTConfig, CIFAR10Config]:
    for latent_dim in config.latent_dims:
        for model in config.models:
            batch_size = 256

            if config.dataset == 'FashionMNIST':
                gpu_mem = '16'
            elif config.dataset == 'CIFAR10':
                gpu_mem = '32'

            if model == "DUL":
                additional_args = "--kl_scale 1e-4"
            elif model == "HIB":
                additional_args = "--kl_scale 1e-4 --K 8"
                batch_size = 64
            else:
                additional_args = ""

            name = f"latentdim_{latent_dim}"
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
                    "num_epoch": config.num_epoch,
                    "additional_args": additional_args,
                    "gpu_mem": gpu_mem
                }
            )

            submit_file = log_dir / 'script.sh'

            with open(submit_file, 'w') as f:
                f.write(submit_script)

            print('Submitting job:', submit_file)
            # Execute code in terminal
            normal = subprocess.run(
                f"bsub < {submit_file}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                shell=True
            )
