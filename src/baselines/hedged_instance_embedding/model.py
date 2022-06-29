# torch stuff
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable


# numpy
import numpy as np

# plotting
import matplotlib.pyplot as plt

BATCH_SIZE=128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load in 2 digit mnist dataset
test = np.load('data/dataset_mnist_2_instance/test.npz')
train = np.load('data/dataset_mnist_2_instance/train.npz')

x_test = torch.flatten(torch.from_numpy(test['images']),start_dim=1).view(-1, 1,28,56).float()
y_test = torch.from_numpy(test['labels'])

x_train = torch.flatten(torch.from_numpy(train['images']),start_dim=1).view(-1, 1,28,56).float()
y_train = torch.from_numpy(train['labels'])

train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1_mu = nn.Conv2d(1, 2, kernel_size=(5,5), stride=2)
        self.conv2_mu = nn.Conv2d(2, 2, kernel_size=(5,5), stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_mu = nn.Linear(10, 2) # 10 is placeholder
        self.fc2_mu = nn.Linear(10, 2) # 10 is placeholder

        self.conv1_scale = nn.Conv2d(1, 2, kernel_size=(5,5), stride=2)
        self.conv2_scale = nn.Conv2d(2, 2, kernel_size=(5,5), stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_scale = nn.Linear(10, 2) # 10 is placeholder
        self.fc2_scale = nn.Linear(10, 2) # 10 is placeholder

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        ### mu ###
        # 1st layer
        mu = self.relu(self.conv1_mu(x))
        mu = self.maxpool(mu)
        mu = self.fc1_mu(mu)

        # 2nd layer
        mu = self.relu(self.conv2_mu(mu))
        mu = self.maxpool(mu)
        mu = self.fc2_mu(mu)
        
        ### scale ###
        # 1st layer
        scale = self.relu(self.conv1_scale(x))
        scale = self.maxpool(scale)
        scale = self.fc1_scale(scale)

        # 2nd layer
        scale = self.relu(self.conv2_scale(scale))
        scale = self.maxpool(scale)
        scale = self.fc2_scale(scale)

        return mu, scale
 


for batch, item in enumerate(train_loader):
    images = item[0]
    labels = item[1]
    
    # we need to get all possible pairs
