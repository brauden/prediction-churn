"""
File for models definitions.
"""
import torch
from torch import nn


class NewsFCNN(nn.Module):
    def __init__(self):
        super(NewsFCNN, self).__init__()
        self.fc_sequence = nn.Sequential(
            nn.Linear(58, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        logits = self.fc_sequence(x)
        return logits
