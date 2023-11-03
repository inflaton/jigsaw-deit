import torch
import torch.nn as nn
import torch.nn.functional as F


class JigsawNet(nn.Module):
    def __init__(self, num_classes=50, num_patches=36, num_features=768):
        super(JigsawNet, self).__init__()

        self.fc1 = nn.Linear(num_patches * num_features, 16384)
        self.fc2 = nn.Linear(16384, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)  # Batch normalization after fc4

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(
            self.bn3(self.fc3(x))
        )  # Apply batch normalization after fc4 and before activation

        return x
