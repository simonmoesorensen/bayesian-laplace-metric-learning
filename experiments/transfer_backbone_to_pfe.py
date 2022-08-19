"""
Script that transfers all Backbone models to PFE /pretrained/ folder
"""
from pathlib import Path
import shutil

src = Path("src")
outputs = Path("outputs")
backbone_models = outputs / "Backbone" / "checkpoints"

pfe_models = src / "baselines" / "PFE" / "pretrained"

for model in backbone_models.glob("**/*.pth"):
    print("Copying {} to {}".format(model, pfe_models))
    dataset = model.parts[-3]
    name = model.parts[-2]
    file_name = f"{name}.pth"

    # Copy model.pth to PFE folder
    pfe_model = pfe_models / dataset / file_name
    pfe_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(model, pfe_model)


# Print total sum size of all files in pfe pretrained folder in MB
total_size = 0
for file in pfe_models.glob("**/*"):
    total_size += file.stat().st_size

print("Total size of pretrined folder: {:.3f} MB".format(total_size / (1024 * 1024.0)))
