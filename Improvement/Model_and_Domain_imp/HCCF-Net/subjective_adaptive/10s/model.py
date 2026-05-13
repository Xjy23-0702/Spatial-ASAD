import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
import math

class CNN_CSP(nn.Module):
    def __init__(self, n_channels=64, csp_components=3):
        super(CNN_CSP, self).__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=n_channels, 
            out_channels=32, 
            kernel_size=(1, 17),
            padding=(0, 8)
        )
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        
        self.register_buffer('csp_filters', torch.zeros(2*csp_components, n_channels))
        self.log_eps = 1e-8
        
        self.fc = nn.Linear(32 + 2*csp_components, 2)
    
    def set_csp_filters(self, filters):
        self.csp_filters.data = torch.tensor(filters, dtype=torch.float32, device=self.csp_filters.device)
    
    def forward(self, x):
        x_cnn = x.permute(0, 2, 1).unsqueeze(-1)  # (B, C, T, 1)
        x_cnn = self.conv_layer(x_cnn)             # (B, 32, T, 1)
        x_cnn = self.bn(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.avg_pool(x_cnn)               # (B, 32, 1, 1)
        cnn_feat = x_cnn.squeeze(-1).squeeze(-1)   # (B, 32)
        cnn_feat = self.dropout(cnn_feat)
        
        x_csp = torch.einsum('btc,fc->btf', x, self.csp_filters)# (B, T, 6)
        var = torch.var(x_csp, dim=1)              # (B, 6)
        var = var / (torch.sum(var, dim=1, keepdim=True) + self.log_eps)
        csp_feat = torch.log(var + self.log_eps)   # (B, 6)
        combined = torch.cat([cnn_feat, csp_feat], dim=1)  # (B, 38)
        out = self.fc(combined)
        
        return out