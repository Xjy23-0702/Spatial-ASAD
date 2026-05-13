import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
from csp_riemannian_layers import SimplifiedCSPRiemannianCNN, CovarianceEnhancedCNN


class CNN_CSP_Riemannian(nn.Module):
    """CSP+Riemannian+CNN混合模型（简化版）"""
    def __init__(self, n_channels=64, n_csp_filters=6):
        super(CNN_CSP_Riemannian, self).__init__()
        self.model = SimplifiedCSPRiemannianCNN(
            n_channels=n_channels,
            n_csp_filters=n_csp_filters
        )
    
    def forward(self, x):
        return self.model(x)


class CNN_CovarianceEnhanced(nn.Module):
    """轻量级协方差增强CNN"""
    def __init__(self, n_channels=64):
        super(CNN_CovarianceEnhanced, self).__init__()
        self.model = CovarianceEnhancedCNN(n_channels=n_channels)
    
    def forward(self, x):
        return self.model(x)