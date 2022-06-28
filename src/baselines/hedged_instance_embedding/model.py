import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

a = np.load('data/dataset_mnist_2_instance/test.npz')

for file in a.files:
    print(file)
    print(a[file].shape)