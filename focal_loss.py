"""
Focal Loss implementation for handling class imbalance in seizure prediction
Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for each class (tensor or float)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class labels
        """
        # Get probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Apply focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combination of Focal Loss and standard Cross Entropy
    """
    def __init__(self, alpha=0.25, gamma=2.0, lambda_focal=0.7):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.ce = nn.CrossEntropyLoss()
        self.lambda_focal = lambda_focal
    
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        ce_loss = self.ce(inputs, targets)
        return self.lambda_focal * focal_loss + (1 - self.lambda_focal) * ce_loss
