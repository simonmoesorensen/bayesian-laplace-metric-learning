"""
Script that transfers all Backbone models to PFE /pretrained/ folder
"""
from pathlib import Path
import shutil

src = Path("src")
outputs = Path("outputs")
backbone_models = outputs / "Backbone" / "checkpoints"

pfe_models = src / "baselines" / "PFE" / "pretrained"

models_dir = [
    path
    for path in backbone_models.glob("**/*")
    if "latentdim" in path.name and path.is_dir()
]

for model in models_dir:
    # Get the most recent checkpoint
    checkpoints = [path for path in model.glob("*.pth")]
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    checkpoint = checkpoints[0]

    # Setup file name
    dataset = checkpoint.parts[-3]
    name = checkpoint.parts[-2]
    file_name = f"{name}.pth"

    # Copy model.pth to PFE folder
    pfe_model = pfe_models / dataset / file_name
    print("Copying {} to {}".format(checkpoint, pfe_model))
    pfe_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkpoint, pfe_model)


# Print total sum size of all files in pfe pretrained folder in MB
total_size = 0
for file in pfe_models.glob("**/*"):
    total_size += file.stat().st_size

print("Total size of pretrined folder: {:.3f} MB".format(total_size / (1024 * 1024.0)))
