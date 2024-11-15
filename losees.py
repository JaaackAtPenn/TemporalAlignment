import torch
import torch.nn as nn
import torch.nn.functional as F

def alignment_classification_loss(pred, target):
    """
    Computes classification loss for alignment prediction.
    
    Args:
        pred (torch.Tensor): Model predictions (B, num_classes)
        target (torch.Tensor): Ground truth labels (B,)
        
    Returns:
        torch.Tensor: Classification loss value
    """
    criterion = nn.CrossEntropyLoss()
    return criterion(pred, target)

def alignment_regression_loss(pred, target):
    """
    Computes regression loss for alignment prediction.
    
    Args:
        pred (torch.Tensor): Model predictions (B, 1)
        target (torch.Tensor): Ground truth values (B, 1)
        
    Returns:
        torch.Tensor: Regression loss value
    """
    criterion = nn.MSELoss()
    return criterion(pred, target)
