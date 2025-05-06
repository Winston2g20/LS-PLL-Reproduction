'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-07 00:05:12
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-07 00:07:44
FilePath: /LS-PLL-Reproduction/codes/ResNet18.py
Description: ResNet18 model implementation.
'''

import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    name = 'ResNet18'
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.resnet(x)