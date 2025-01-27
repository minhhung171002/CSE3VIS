import torch
import torch.nn as nn
import torchvision
from torchvision import models


# TODO Task 1c - Implement a SimpleBNConv
class SimpleBNConv(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleBNConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(128*7*7, num_classes)
    def forward(self, x):
      x = self.features(x)
      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      return x

# TODO Task 1f - Create a model from a pre-trained model from the torchvision
#  model zoo.
# Load a pre-trained ResNet-18 model
pretrained_resnet18 = models.resnet18(pretrained=True)

# Modify the model for fine-tuning (unfreeze all layers)
for param in pretrained_resnet18.parameters():
    param.requires_grad = True

# Replace the output layer with a new classification layer
num_classes = 7  
in_features = pretrained_resnet18.fc.in_features
pretrained_resnet18.fc = nn.Linear(in_features, num_classes)

# TODO Task 1f - Create your own models
class CustomModel(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomModel, self).__init__()
        # Define your custom architecture here
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more layers as needed
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

