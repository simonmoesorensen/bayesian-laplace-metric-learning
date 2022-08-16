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
                num_epoch = 150
            elif config.dataset == 'CIFAR10':
                num_epoch = 500

            if model == "DUL":
                additional_args = "--kl_scale 1e-4"
            elif model == "HIB":
                additional_args = "--kl_scale 1e-4"
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
                    "additional_args": additional_args,
                    "num_epoch": num_epoch
                }
            )

            submit_file = log_dir / 'script.sh'

            with open(submit_file, 'w') as f:
                f.write(submit_script)

            normal = subprocess.run(
                f"bsub < {submit_file}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                # text=True,
                shell=True
            )
            break
            # Execute code in terminal


print(template_text)