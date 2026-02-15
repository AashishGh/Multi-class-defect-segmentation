import torch
import torch.nn as nn


class FocalTverskyLossFG(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.0, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)          # [B,1,H,W]
        y = targets                        # [B,1,H,W] in {0,1}
        TP = (p * y).sum()
        FP = ((1 - y) * p).sum()
        FN = (y * (1 - p)).sum()
        t = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return (1 - t) ** self.gamma