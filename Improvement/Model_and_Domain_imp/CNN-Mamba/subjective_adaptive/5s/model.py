import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
import math
from mamba_ssm import Mamba

# --------------------------CNN-baseline-----------------------------
class CNN_Mamba(nn.Module):
    def __init__(self):
        super().__init__()

        self.temporal_conv = nn.Conv2d(1, 16, (9,1), padding=(4,0))
        self.spatial_conv = nn.Conv2d(16, 32, (1,64))

        self.relu = nn.ReLU()

        # Mamba expects (B, T, C)
        self.mamba = Mamba(
            d_model=32,
            d_state=64,
            d_conv=4,
            expand=2
        )

        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,T,C)

        x = self.relu(self.temporal_conv(x))   # (B,16,T,C)
        x = self.relu(self.spatial_conv(x))    # (B,32,T,1)

        x = x.squeeze(-1)                      # (B,32,T)
        x = x.permute(0,2,1)                   # (B,T,32)

        x = self.mamba(x)                      # (B,T,32)

        x = x.mean(dim=1)                      # temporal pooling

        out = self.fc(x)
        return out








