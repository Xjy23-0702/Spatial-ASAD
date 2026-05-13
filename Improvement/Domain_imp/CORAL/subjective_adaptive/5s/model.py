import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
import math


# --------------------------CNN-baseline-----------------------------
class CNN_baseline(nn.Module):
    def __init__(self):
        super(CNN_baseline, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(17,64), padding=(8, 0))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(cfg.decision_window, 1))
        self.fc1 = nn.Linear(in_features=5, out_features=5)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=5, out_features=2)

    def get_features(self, x):
        x = x.unsqueeze(dim=1)
        conv_out = self.conv_layer(x)
        relu_out = self.relu(conv_out)
        avg_pool_out = self.avg_pool(relu_out)
        features = torch.flatten(avg_pool_out, start_dim=1)
        return features

    def forward(self, x):
        features = self.get_features(x)
        fc1_out = self.fc1(features)
        sigmoid_out = self.sigmoid(fc1_out)
        fc2_out = self.fc2(sigmoid_out)
        return fc2_out



