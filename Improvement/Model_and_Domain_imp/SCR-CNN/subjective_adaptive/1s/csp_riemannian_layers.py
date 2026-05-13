import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentiableCSPLayer(nn.Module):
    """
    可微分CSP层：将多通道EEG信号投影到最优空间滤波器
    输入: (batch, channels, time)
    输出: (batch, n_filters, time)
    """
    def __init__(self, n_channels=64, n_filters=6, kernel_size=17):
        super(DifferentiableCSPLayer, self).__init__()
        
        self.n_channels = n_channels
        self.n_filters = n_filters
        
        # 使用可学习的空间滤波器
        self.spatial_filters = nn.Parameter(
            torch.randn(n_filters, n_channels) * 0.1
        )
        
        # 时间卷积
        self.temporal_conv = nn.Conv1d(
            n_filters, n_filters, kernel_size, 
            padding=kernel_size//2, groups=n_filters, bias=False
        )
        #self.bn = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        batch, channels, time = x.shape
        # 空间滤波
        spatial_filtered = torch.einsum('bct,fc->bft', x, self.spatial_filters)
        # 时间卷积
        spatial_filtered = self.temporal_conv(spatial_filtered)
        #spatial_filtered = self.bn(spatial_filtered)
        spatial_filtered = self.relu(spatial_filtered)
        return spatial_filtered


class SimplifiedCSPRiemannianCNN(nn.Module):
    """
    简化版CSP+Riemannian+CNN混合模型（修复维度问题）
    """
    def __init__(self, n_channels=64, n_csp_filters=6, time_length=128):
        super(SimplifiedCSPRiemannianCNN, self).__init__()
        
        self.n_channels = n_channels
        self.n_csp_filters = n_csp_filters
        
        # CSP层
        self.csp_layer = DifferentiableCSPLayer(
            n_channels=n_channels,
            n_filters=n_csp_filters,
            kernel_size=17
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # CSP特征投影层
        self.csp_proj = nn.Sequential(
            nn.Linear(n_csp_filters, 16),
            nn.ReLU(),
            #nn.Dropout(0.3)
        )
        
        # 原始CNN分支
        # 输入: (batch, 1, time, channels)
        # Conv2d: 1->8, kernel=(17, n_channels) -> 输出 (batch, 8, time, 1)
        self.cnn_conv = nn.Conv2d(1, 8, kernel_size=(17, n_channels), padding=(8, 0))
        self.cnn_relu = nn.ReLU()
        # AdaptiveAvgPool2d((1, 1)) -> (batch, 8, 1, 1)
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_proj = nn.Sequential(
            nn.Flatten(),  # (batch, 8)
            nn.Linear(8, 16),
            nn.ReLU(),
            #nn.Dropout(0.3)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(16, 2)
        )
        
        print(f"SimplifiedCSPRiemannianCNN initialized:")
        print(f"  - CSP filters: {n_csp_filters}")
        print(f"  - CNN output: 16, CSP output: 16, Fusion input: 32")
    
    def forward(self, x):
        # x: (batch, time, channels)
        if x.dim() == 4:
            x = x.squeeze(1)
        
        batch, time, channels = x.shape
        
        # CNN分支
        cnn_in = x.unsqueeze(1)  # (batch, 1, time, channels)
        cnn_conv_out = self.cnn_conv(cnn_in)  # (batch, 8, time, 1)
        cnn_relu_out = self.cnn_relu(cnn_conv_out)  # (batch, 8, time, 1)
        cnn_pool_out = self.cnn_pool(cnn_relu_out)  # (batch, 8, 1, 1)
        cnn_features = self.cnn_proj(cnn_pool_out)  # (batch, 16)
        
        # CSP分支
        csp_in = x.permute(0, 2, 1)  # (batch, channels, time)
        csp_out = self.csp_layer(csp_in)  # (batch, n_filters, time)
        csp_pooled = self.global_avg_pool(csp_out).squeeze(-1)  # (batch, n_filters)
        csp_features = self.csp_proj(csp_pooled)  # (batch, 16)
        
        # 融合
        fused = torch.cat([cnn_features, csp_features], dim=1)  # (batch, 32)
        output = self.fusion(fused)  # (batch, 2)
        
        return output

class CovarianceEnhancedCNN(nn.Module):
    """
    轻量级协方差增强CNN（修复维度问题）
    """
    def __init__(self, n_channels=64):
        super(CovarianceEnhancedCNN, self).__init__()
        
        self.n_channels = n_channels
        
        # 原始CNN baseline风格
        # 输入: (batch, 1, time, channels) where time=128, channels=64
        # Conv2d: (1, 5, kernel=(17,64)) -> 输出 (batch, 5, time-17+1+16, 1) 
        # 由于padding=(8,0)，时间维度不变: 128 -> 128
        self.conv_layer = nn.Conv2d(1, 5, kernel_size=(17, n_channels), padding=(8, 0))
        self.relu = nn.ReLU()
        # AdaptiveAvgPool2d((5, 1)) -> (batch, 5, 5, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((5, 1))
        
        # 计算CNN分支的输出维度
        # 经过 avg_pool 后: (batch, 5, 5, 1) -> flatten -> (batch, 25)
        self.cnn_out_dim = 5 * 5  # = 25
        
        # 特征增强分支（使用1D卷积提取时域特征）
        # 输入: (batch, channels, time) = (batch, 64, 128)
        self.enhance_branch = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=17, padding=8),  # (batch, 16, 128)
            #nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (batch, 16, 1)
            nn.Flatten(),  # (batch, 16)
            nn.Linear(16, 10),
            nn.ReLU()
        )
        self.enhance_out_dim = 10
        
        # 融合分类器
        self.fusion_in_dim = self.cnn_out_dim + self.enhance_out_dim  # 25 + 10 = 35
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_in_dim, 16),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(16, 2)
        )
        
        print(f"CovarianceEnhancedCNN initialized:")
        print(f"  - CNN out dim: {self.cnn_out_dim}")
        print(f"  - Enhance out dim: {self.enhance_out_dim}")
        print(f"  - Fusion input dim: {self.fusion_in_dim}")
    
    def forward(self, x):
        # x: (batch, time, channels)
        if x.dim() == 4:
            x = x.squeeze(1)
        
        batch, time, channels = x.shape
        
        # CNN主分支
        cnn_in = x.unsqueeze(1)  # (batch, 1, time, channels)
        conv_out = self.conv_layer(cnn_in)  # (batch, 5, time, 1)
        relu_out = self.relu(conv_out)  # (batch, 5, time, 1)
        avg_out = self.avg_pool(relu_out)  # (batch, 5, 5, 1)
        cnn_features = avg_out.flatten(1)  # (batch, 25)
        
        # 增强分支
        enhance_in = x.permute(0, 2, 1)  # (batch, channels, time)
        enhance_features = self.enhance_branch(enhance_in)  # (batch, 10)
        
        # 融合
        fused = torch.cat([cnn_features, enhance_features], dim=1)  # (batch, 35)
        output = self.classifier(fused)  # (batch, 2)
        
        return output