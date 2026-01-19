import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. The missing BCE Criterion ---
# We use BCEWithLogitsLoss because your UNet output is likely raw logits
# (based on the fact that you apply torch.sigmoid manually before dice_loss in train_segmentation.py)
bce_criterion = nn.BCEWithLogitsLoss()


# --- 2. The missing Dice Loss ---
def dice_loss(pred, target, smooth=1e-6):
    """
    Standard Dice Loss for segmentation.
    Args:
        pred: Predicted probabilities (B, C, H, W). Ensure sigmoid is applied if model outputs logits.
        target: Ground truth masks (B, C, H, W).
    """
    # Flatten tensors to simplify calculation
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()

    # Dice coefficient formula
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice

def boundary_loss(pred, target, boundary_weight=5.0):
    """
    Args:
        pred: Predicted tensor (B, C, H, W) or (B, C, D, H, W)
        target: Ground truth tensor (same shape as pred)
        boundary_weight: Weighting factor for the loss
    """
    max_pool=0
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