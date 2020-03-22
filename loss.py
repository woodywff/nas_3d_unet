import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb

class WeightedDiceLoss(nn.Module):
    def __init__(self, axis=(-1,-2,-3), smooth=1e-6):
        super().__init__()
        self.axis = axis
        self.smooth = smooth
        
    def forward(self, y_pred, y_truth):
        return 1 - torch.mean((2 * torch.sum(y_pred * y_truth, dim = self.axis) + self.smooth)/
                         (torch.sum(y_pred, dim = self.axis) + torch.sum(y_truth, dim = self.axis) + self.smooth))