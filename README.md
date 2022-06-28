# Baselines

| folder                          | paper                                                                                                      | code                                                                                                           |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `face_recognition`              | [Data Uncertainty Learning in Face Recognition](https://arxiv.org/pdf/2003.11339.pdf)                      | [Ontheway361](https://github.com/Ontheway361/dul-pytorch), [MouxiaoHuang](https://github.com/MouxiaoHuang/DUL) |
| `hedged_instance_embedding`     | [MODELING UNCERTAINTY WITH HEDGED INSTANCE EMBEDDING](https://arxiv.org/pdf/1810.00319.pdf)                | [RRoundTable](https://github.com/RRoundTable/hedged_instance_embedding)                                        |
| `probabilistic_face_embedding`  | [Probabilistic Face Embeddings](https://arxiv.org/pdf/1904.09658.pdf)                                      | [seasonSH](https://github.com/seasonSH/Probabilistic-Face-Embeddings)                                          |
| `unsupervised_visual_retrieval` | [Unsupervised Data Uncertainty Learning in Visual Retrieval Systems](https://arxiv.org/pdf/1902.02586.pdf) |                                                                                                                |


To make n-digit MNIST dataset, you need to have a data directory, and then run `download.sh` to get the mnist dataset. Make sure you also have downloaded the `standard_datasets` directory from https://github.com/google/n-digit-mnist. Then, run `n_digit_mnist.py` with `python n_digit_mnist.py --num_digits 2 --domain_gap instance --use_standard_dataset` to make the 2 digit mnist dataset in `data/dataset_mnist_2_instance` directory. 