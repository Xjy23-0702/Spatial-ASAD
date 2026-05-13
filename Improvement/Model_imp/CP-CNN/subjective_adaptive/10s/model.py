import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
import math

class CNN_CSP_Projection(nn.Module):
    def __init__(self, n_channels=64, csp_components=3):
        super(CNN_CSP_Projection, self).__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=2*csp_components, 
            out_channels=32, 
            kernel_size=(1, 17),
            padding=(0, 8)
        )
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32, 2)
    
    def forward(self, x):
        # x: (B, T, C)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, 6, T, 1)
        x = self.conv_layer(x)                 # (B, 32, T, 1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)                   # (B, 32, 1, 1)
        x = x.squeeze(-1).squeeze(-1)          # (B, 32)
        x = self.dropout(x)
        return self.fc(x)