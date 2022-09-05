# Baselines

| name                          | paper                                                                                                      | code                                                                                                           |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `deep_uncertainty_learning (DUL)`              | [Data Uncertainty Learning in Face Recognition](https://arxiv.org/pdf/2003.11339.pdf)                      | [Ontheway361](https://github.com/Ontheway361/dul-pytorch), [MouxiaoHuang](https://github.com/MouxiaoHuang/DUL) |
| `hedged_instance_embedding (HIB)`     | [MODELING UNCERTAINTY WITH HEDGED INSTANCE EMBEDDING](https://arxiv.org/pdf/1810.00319.pdf)                | [RRoundTable](https://github.com/RRoundTable/hedged_instance_embedding)                                        |
| `probabilistic_face_embedding (PFE)`  | [Probabilistic Face Embeddings](https://arxiv.org/pdf/1904.09658.pdf)                                      | [seasonSH](https://github.com/seasonSH/Probabilistic-Face-Embeddings)                         |

# Getting started
In this section we will cover how to setup the environment and run the models

## Setup
1. In this project we use python 3.8.11 and CUDA 11.7, so run `module load python3/3.8.11; module load cuda/11.7`
2. Create and activate the environment with `python3 -m venv venv/; source venv/bin/activate`
3. `python3 -m pip install --no-cache-dir -U pip; python3 -m pip install --no-cache-dir -r requirements.txt`
4. Create a file named `.env` in the root directory and fill it with the following line: `DATA_DIR=/work3/XXXXXX/datasets/`, where `XXXXXX` is your DTU ID.

## Running the code
The code for training is contained in the `train.py` files located in `src/baselines/<BASELINE>/train.py` which is accompanied by a `config.py` in the same directory which contains all necessary arguments for the code.

> For laplace, the train script is in `src/laplace/train_post_hoc.py`

Examples of run-scripts can be found in `scripts/<BASELINE>/<DATASET>/train.sh` which you can easily run in a bash terminal. 

If you wish to run the `train.py` manually, you can run:

```
python3 <path_to_train.py>
```

## Code architecture

As you can see in the `train.py` we have structured the code around two core principles:

1. DataModules
2. LightningModules

### DataModules
A datamodule is what defines which dataset we train, validate and test on. In addition to the usual definitions, we also include the OOD dataset in each datamodule such that we can compare in distribution versus out of distribution.

### LightningModules
We use a lite version of pytorch-lightning that enables us to easily scale across multiple gpus whilst maintaining full control over the train loops and steps of each model. We implement a `train_step`, `val_step`, `test_step` and `ood_step` for all models that enable us to reuse code across models, similar to pytorch-lightning. For post-hoc laplace, the training loop is only a single epoch and it doesn't use an optimizer, and is therefore overwritten to accomodate for this. 

From a visualization perspective, we visualize every given `freq` if the `to_visualize` argument is set (specified in the config or CLI). 
At every visualization step, we visualize all plots (top/bottom 5, id vs ood, roc curve, prc curve, sparsification curve, calibration curve) and generate metrics in .json files.

## Scaling experiments

Under the `experiments` folder, we have scripts that can run all experiments systematically and seamlessly, assuming that they are run in the DTU HPC cluster (or any LSF-based cluster). The steps are as follows:

1. Setup your config in `config.py`
2. Train backbone models with `submit_experiments_backbone.py` if not existent in `src/baselines/PFE/pretrained/*`
    1. Move backbones to PFE with `transfer_backbone_to_pfe.py`
3. Train baselines with `submit_experiments.py`
4. Train laplace with `submit_experiments_laplace.py`
5. When all experiments are finished, gather them by running `gather_experiments.py`

# Debugging in HPC

First, install the `debugpy` package:
```bash
pip install debugpy
```

Add this configuration to your `.vscode/launch.json` file

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "10.66.20.1",
                "port": <YOUR_PORT>
            }
        }
    ]
}
```

Open an interactive terminal using `qrsh` (cpu) or `voltash` (gpu)

Run your python scripts like this:

```bash
python -m debugpy --wait-for-client --listen 10.66.20.1:<YOUR_PORT> <script.py>
```

> Hint: The "host" is the IP address of the node you are debugging. You can view the ip using `ifconfig` in your terminal. It should be the top most print under the "eth0" keyword and then "inet"
