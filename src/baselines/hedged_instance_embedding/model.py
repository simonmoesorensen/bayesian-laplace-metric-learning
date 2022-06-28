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

BATCH_SIZE=200

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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 
cnn = CNN()

