"""
This script gathers all the experiments in the outputs/ directory and writes them to a 
csv file.
"""

import json
from pathlib import Path
import pandas as pd
from config import CIFAR10Config, FashionMNISTConfig

# Find all figure folders in outputs/
outputs_dir = Path("outputs")

experiments = [
    path
    for path in outputs_dir.glob("**/*")
    if path.is_dir()
    and (path / "hparams.json").exists()
    and ("latentdim" in path.parent.name)
]

records = []
# Read all experiments data
for path in experiments:
    print("Loading experiment from {}".format(path))
    # Read the hparams.json file
    hparams = json.load(open(path / "hparams.json"))

    # Read the metrics.json file
    metrics = json.load(open(path / "metrics.json"))

    # Read the additional metrics.json file
    additional_metrics = json.load(open(path / "additional_metrics.json"))

    data = {
        "seed": hparams["random_seed"],
        "model_name": path.parent.parent.parent.parent.name,
        "dataset": hparams["dataset"],
        "latent_dim": hparams["embedding_size"],
        "l2_norm": "yes",
        "acc": metrics["test_accuracy"],
        "map@5": metrics["test_map_k"],
        "recall@5": metrics["test_recall_k"],
        "auroc": additional_metrics["test_auroc"],
        "auprc": additional_metrics["test_auprc"],
        "ausc": additional_metrics["test_ausc"],
        "ece": additional_metrics["test_ece"],
    }

    records.append(data)

df = pd.DataFrame.from_records(records)

df_grouped = df.groupby(["model_name", "dataset", "latent_dim"]).apply(
    lambda x: x.loc[x.acc.idxmax()]
)
df_grouped["success_rate"] = (
    df.groupby(["model_name", "dataset", "latent_dim"]).count()["seed"] / 5
)

# If a run is missing, add 0 success_rate and 0 to all metrics
missing_data = []
for config in [FashionMNISTConfig, CIFAR10Config]:
    for latent_dim in config.latent_dims:
        for model in config.models:
            if (model, config.dataset, latent_dim) not in df_grouped.index:
                missing_data.append(
                    {
                        "model_name": model,
                        "dataset": config.dataset,
                        "latent_dim": latent_dim,
                        "success_rate": 0,
                        "l2_norm": "N/A",
                        "acc": 0,
                        "map@5": 0,
                        "recall@5": 0,
                        "auroc": 0,
                        "auprc": 0,
                        "ausc": 0,
                        "ece": 0,
                    }
                )

df_grouped = df_grouped.append(missing_data)
print(df_grouped)
df_grouped.sort_values(["dataset", "model_name", "latent_dim"], inplace=True)

for config in [FashionMNISTConfig, CIFAR10Config]:
    df_save = df_grouped.query("dataset == @config.dataset")

    df_save.to_csv(
        outputs_dir / f"experiments_{config.dataset}.csv",
        index=False,
        float_format="%.4f",
        encoding="utf-8",
        sep=",",
        decimal=".",
    )
