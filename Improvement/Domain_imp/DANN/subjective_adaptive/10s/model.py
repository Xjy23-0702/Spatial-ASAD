import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时乘以负的alpha，实现梯度反转
        return grad_output.neg() * ctx.alpha, None

class CNN_baseline(nn.Module):
    def __init__(self):
        super(CNN_baseline, self).__init__()
        
        # 特征提取器（与你的CNN_baseline相同）
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(17,64), padding=(8, 0)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(cfg.decision_window, 1))
        )
        
        # 标签分类器（左右耳分类）
        self.label_classifier = nn.Sequential(
            nn.Linear(in_features=5, out_features=5),
            nn.Sigmoid(),
            nn.Linear(in_features=5, out_features=2)
        )
        
        # 域判别器（判断特征来自哪个被试/域）
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features=5, out_features=32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 二分类：源域(0) vs 目标域(1)
        )
    
    def forward(self, x, alpha=1.0, return_domain=False):
        # 输入形状: (batch, time, channels)
        x = x.unsqueeze(dim=1)  # (batch, 1, time, channels)
        
        # 特征提取
        features = self.feature_extractor(x)  # (batch, 5, 1, 1)
        features = torch.flatten(features, start_dim=1)  # (batch, 5)
        
        # 标签分类
        label_out = self.label_classifier(features)
        
        if return_domain:
            # 域分类（带梯度反转）
            reversed_features = GradientReversalLayer.apply(features, alpha)
            domain_out = self.domain_classifier(reversed_features)
            return label_out, domain_out
        
        return label_out