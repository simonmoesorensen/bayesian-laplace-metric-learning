from config import FashionMNISTConfig, CIFAR10Config, template_text
import subprocess
from pathlib import Path

root = Path(__file__).parent.parent

model = 'Backbone'
# Submit FashionMNIST experiments
for config in [FashionMNISTConfig, CIFAR10Config]:
    for latent_dim in config.latent_dims:
        batch_size = 512

        if config.dataset == 'FashionMNIST':
            num_epoch = 100
        elif config.dataset == 'CIFAR10':
            num_epoch = 500

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
                "num_epoch": num_epoch,
                "additional_args": "",
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
