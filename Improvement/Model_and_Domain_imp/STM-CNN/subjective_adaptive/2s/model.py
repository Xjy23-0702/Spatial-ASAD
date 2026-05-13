import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
import math


# --------------------------CNN-baseline-----------------------------
class CNN_baseline(nn.Module):
    def __init__(self):
        super(CNN_baseline, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 17),  padding=(0, 8))
        #self.bn_conv = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(cfg.decision_window, 1))
        #self.dropout = nn.Dropout(dropout_rate) 
        self.fc1 = nn.Linear(in_features=32, out_features=5)
        #self.bn_fc = nn.BatchNorm1d(5) 
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=5, out_features=2)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(-1)
        conv_out = self.conv_layer(x)
        relu_out = self.relu(conv_out)
        avg_pool_out = self.avg_pool(relu_out)
        flatten_out = torch.flatten(avg_pool_out, start_dim=1)
        fc1_out = self.fc1(flatten_out)
        sigmoid_out = self.sigmoid(fc1_out)
        fc2_out = self.fc2(sigmoid_out)

        return fc2_out








