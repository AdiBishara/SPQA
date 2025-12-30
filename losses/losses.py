import torch
import torch.nn as nn
import torch.nn.functional as F

def boundary_loss(pred, target, boundary_weight=5.0):
    """
    Args:
        pred: Predicted tensor (B, C, H, W) or (B, C, D, H, W)
        target: Ground truth tensor (same shape as pred)
        boundary_weight: Weighting factor for the loss
    """
    target = target.float()
    if pred.dim() == 4:
        max_pool = F.max_pool2d
    elif pred.dim() == 5:
        max_pool = F.max_pool3d
    # Create the Boundary Map
    target_dilated = max_pool(target, kernel_size=3, stride=1, padding=1)

    target_eroded = -max_pool(-target, kernel_size=3, stride=1, padding=1)
    
    boundary_map = target_dilated - target_eroded

    # Calculate Loss
    diff = torch.abs(pred - target)
    
    # Weighted loss focused on boundaries
    loss = (diff * boundary_map).mean()

    return boundary_weight * loss