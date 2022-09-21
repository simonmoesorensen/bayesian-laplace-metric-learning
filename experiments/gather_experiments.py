"""
This script gathers all the experiments in the outputs/ directory and writes them to a 
csv file.
"""

import json
from pathlib import Path
import pandas as pd
from config import CIFAR10Config, FashionMNISTConfig
import shutil

# Find all figure folders in outputs/
outputs_dir = Path("outputs")
results_dir = Path("results")
github_prefix = (
    "https://github.com/simonmoesorensen/bayesian-laplace-metric-learning/tree/main/"
)

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

    # Read the expected_metrics.json file
    expected_metrics = json.load(open(path / "expected_metrics.json"))

    data = {
        "seed": hparams["random_seed"],
        "model_name": path.parent.parent.parent.parent.name,
        "dataset": hparams["dataset"],
        "latent_dim": hparams["embedding_size"],
        "l2_norm": "yes",
        "acc": metrics["test/accuracy"],
        "map@5": metrics["test/map_k"],
        "recall@5": metrics["test/recall_k"],
        "auroc": additional_metrics["test/auroc"],
        "auprc": additional_metrics["test/auprc"],
        "ausc": additional_metrics["test/ausc"],
        "ece": additional_metrics["test/ece"],
        "expected_acc": expected_metrics["test_expected/accuracy"],
        "expected_map@5": expected_metrics["test_expected/map_k"],
        "expected_recall@5": expected_metrics["test_expected/recall_k"],
        "path": path,
    }

    records.append(data)

df = pd.DataFrame.from_records(records)


# Group over all seeds (choose best seed for visualization purposes)
df_grouped = df.groupby(["model_name", "dataset", "latent_dim"]).apply(
    lambda x: x.loc[x.acc.idxmax()]
)

# Calculate mean and std of metrics
metrics = [
    "acc",
    "map@5",
    "recall@5",
    "auroc",
    "auprc",
    "ausc",
    "ece",
    "expected_acc",
    "expected_map@5",
    "expected_recall@5",
]

df_stats = (
    df[metrics + ["model_name", "dataset", "latent_dim"]]
    .groupby(["model_name", "dataset", "latent_dim"])
    .agg(["mean", "std"])
)

# Join mean and std to one column with format: metric +- std
for metric in metrics:
    df_grouped[metric] = (
        df_stats[metric]["mean"].round(3).astype(str)
        + " ± "
        + df_stats[metric]["std"].round(3).astype(str)
    )


# Remove results folder if exists
if results_dir.exists():
    shutil.rmtree(results_dir)

# Copy best seed figures to results folder
best_paths = df_grouped.path.values
df_grouped = df_grouped.drop(columns=["path"])

figure_paths = []
for path in best_paths:
    print("Copying {} to {}".format(path, results_dir))
    # Remove 'outputs' from the path
    new_path = Path(*path.parts[1:])
    new_path = results_dir / new_path
    new_path.mkdir(parents=True, exist_ok=True)

    # Only copy .png files
    for file in path.glob("**/*.png"):
        shutil.copy(file, new_path)

    figure_paths.append(github_prefix + str(new_path))

# Add figure paths to the dataframe
df_grouped["figure_path"] = figure_paths

# Add success rate of runs
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
                        "expected_acc": 0,
                        "expected_map@5": 0,
                        "expected_recall@5": 0,
                    }
                )

df_grouped = pd.concat([df_grouped, pd.DataFrame.from_records(missing_data)])

# Sort values
df_grouped.sort_values(["dataset", "latent_dim", "model_name"], inplace=True)

# Set column order to match the table in the paper
df_grouped = df_grouped[
    [
        "dataset",
        "seed",
        "model_name",
        "latent_dim",
        "l2_norm",
        "figure_path",
        "acc",
        "map@5",
        "recall@5",
        "auroc",
        "auprc",
        "ausc",
        "ece",
        "expected_acc",
        "expected_map@5",
        "expected_recall@5",
        "success_rate",
    ]
]

print(df_grouped)
# Save results per dataset
for config in [FashionMNISTConfig, CIFAR10Config]:
    df_save = df_grouped.query("dataset == @config.dataset")
    df_save = df_save.drop(columns=["dataset"])
    df_save.to_csv(
        results_dir / f"experiments_{config.dataset}.csv",
        index=False,
        float_format="%.5f",
        encoding="utf-8",
        sep=",",
        decimal=".",
    )

# Print total sum size of all files in results folder in MB
total_size = 0
for file in results_dir.glob("**/*"):
    total_size += file.stat().st_size

print("Total size of results folder: {:.3f} MB".format(total_size / (1024 * 1024.0)))
