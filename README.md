# Baselines

| folder                          | paper                                                                                                      | code                                                                                                           |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `deep_uncertainty_learning`              | [Data Uncertainty Learning in Face Recognition](https://arxiv.org/pdf/2003.11339.pdf)                      | [Ontheway361](https://github.com/Ontheway361/dul-pytorch), [MouxiaoHuang](https://github.com/MouxiaoHuang/DUL) |
| `hedged_instance_embedding`     | [MODELING UNCERTAINTY WITH HEDGED INSTANCE EMBEDDING](https://arxiv.org/pdf/1810.00319.pdf)                | [RRoundTable](https://github.com/RRoundTable/hedged_instance_embedding)                                        |
| `probabilistic_face_embedding`  | [Probabilistic Face Embeddings](https://arxiv.org/pdf/1904.09658.pdf)                                      | [seasonSH](https://github.com/seasonSH/Probabilistic-Face-Embeddings)                                          |
| `unsupervised_visual_retrieval` | [Unsupervised Data Uncertainty Learning in Visual Retrieval Systems](https://arxiv.org/pdf/1902.02586.pdf) |                                                                                                                |

# Getting started

* In this project we use python 3.8.11 and CUDA 11.7, so run `module load python3/3.8.11; module load cuda/11.7`
* Create and activate the environment with `python3 -m venv venv/; source venv/bin/activate`
* `python3 -m pip install --no-cache-dir -U pip; python3 -m pip install --no-cache-dir -r requirements.txt`
* Create a file named `.env` in the root directory and fill it with the following line: `DATA_DIR=/work3/sXXXXXX/datasets/`, where `XXXXXX` is your student ID.

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
