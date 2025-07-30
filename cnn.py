import polars as pl
from dbg import dprint, tstats
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn


# classes = sorted(set(labs['lab'].to_list()))
# dprint(1, f"Classes: {classes[:10]}... Total: {len(classes)}")

class CNN(torch.nn.Module):
    def __init__(self, num_filters=64, kernel_size=19, seq_len=8192,fc_dim=64, num_classes=1313):
        super().__init__()
        # convolutional layer
        self.conv1 = nn.Conv1d( in_channels=5, # 5 channels for A, T, G, C, N
                                out_channels=num_filters, 
                                kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        # fully connected layers
        self.fc1 = nn.Linear(num_filters, fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2]) # global pooling
        x = x.squeeze(-1)
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.fc2(x)
        return x