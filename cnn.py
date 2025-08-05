import polars as pl
from dbg import dprint, tstats
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn


# classes = sorted(set(labs['lab'].to_list()))
# dprint(1, f"Classes: {classes[:10]}... Total: {len(classes)}")

class CNN(torch.nn.Module):
    def __init__(self, num_filters=64, kernel_size=19, seq_len=8192,fc_dim=128, num_classes=1313):
        super().__init__()
        # 5 channels for A, T, G, C, N
        self.conv1 = nn.Conv1d( in_channels=5, 
                                out_channels=num_filters, 
                                kernel_size=kernel_size,
                                padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(in_channels=num_filters,
                               out_channels=num_filters*2,
                               kernel_size=7,
                               padding=3)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        # fully connected layers
        self.fc1 = nn.Linear(num_filters*2, fc_dim)
        self.bn3 = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = F.avg_pool1d(x, x.shape[2]) # global pooling
        x = x.squeeze(-1)
        
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = self.fc2(x)
        return x