import torch
import torch.nn as nn


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size, img_size, n_channels=3):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(n_channels, 16, 3, 1), nn.ReLU(),
                                     nn.MaxPool2d(1, stride=2),
                                     nn.Conv2d(16, 32, 3, 1), nn.ReLU(),
                                     nn.MaxPool2d(1, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.ReLU(),
                                     nn.MaxPool2d(1, stride=2))

        # Calculate output size of convnet
        dummy_input = torch.rand(1, n_channels, img_size, img_size)
        dummy_output = self.convnet(dummy_input)
        output_size = dummy_output.view(1, -1).size(1)

        self.fc = nn.Sequential(nn.Linear(output_size, embedding_size))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
