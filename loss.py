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
        assert y_pred.device == y_truth.device, 'y_pred.device != y_truth.device'
        device = y_pred.device
        y_pred = torch.as_tensor(y_pred, dtype = float, device = device)
        y_truth = torch.as_tensor(y_truth, dtype = float, device = device)
        return - torch.mean((2 * torch.sum(y_pred * y_truth, dim = self.axis) + self.smooth)/
                         (torch.sum(y_pred, dim = self.axis) + torch.sum(y_truth, dim = self.axis) + self.smooth))