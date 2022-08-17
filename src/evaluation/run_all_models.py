"""
Script for running evaluation on all models in the 'models' folder.

The script finds all leaf nodes in the model folder containing a 'model.pth' file and
a 'hparams.json' file.

Example:
    No debugger:
        python3 -m src.evaluation.run_all_models

    With debugger:
        python3 -m debugpy --listen 10.66.12.19:1332 src/evaluation/run_all_models.py

"""

import json
from pathlib import Path
from src.evaluation.calibration_curve import load as run_calibration_curve
from src.evaluation.sparsification_curve import load as run_sparsification_curve

print("Running evaluation for all models in /models")
# root
root = Path(__file__).parent.parent.parent.absolute()

# models
models_dir = root / "models"

# Find all folders with model.pth and hparams.json files
model_dirs = [
    model_dir
    for model_dir in models_dir.glob("**/*")
    if model_dir.is_dir()
    and (model_dir / "model.pth").exists()
    and (model_dir / "hparams.json").exists()
]

# Run evaluation for each model
for model_dir in model_dirs:
    print(f"Running evaluation for {model_dir}")

    hparams = json.load(open(model_dir / "hparams.json"))
    model_path = model_dir / "model.pth"

    # Find parent with name '/models/'
    parent_dir = model_dir.parent
    idx = 0
    while parent_dir.name != "models":
        parent_dir = parent_dir.parent
        idx += 1

    # Get model name from child of models folder
    model = model_dir.parents[idx - 1].name

    # Get parameters from hparams
    dataset = hparams["dataset"]
    embedding_size = hparams["embedding_size"]
    batch_size = hparams["batch_size"]

    if "loss" in hparams and model == "PFE":
        loss = hparams["loss"]
    else:
        loss = None

    print("Running calibration curve")
    run_calibration_curve(
        model_name=model,
        model_path=model_path,
        dataset=dataset,
        embedding_size=embedding_size,
        batch_size=batch_size,
        loss=loss,
        samples=100,
    )

    print("Running sparsification curve")
    run_sparsification_curve(
        model_name=model,
        model_path=model_path,
        dataset=dataset,
        embedding_size=embedding_size,
        batch_size=batch_size,
        loss=loss,
    )
    print(f"Finished evaluation for {model_dir}")
