import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# TODO Task 1c - Implement a SimpleBNConv
#        The SimpleBNConv should have:
#        - 5 nn.Conv2d layers, with 8, 16, 32, 64, 128 output channels respectively -- DONE
#        - nn.ReLU() activation function between each convolution layer
#        - nn.BatchNorm2d between each convolution layer
#        - nn.MaxPool2d to downsample by a factor of 2

class SimpleBNConv(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleBNConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 128 * 7 * 7) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# TODO Task 1f - Create a model from a pre-trained model from the torchvision
#  model zoo.


# TODO Task 1f - Create your own models

