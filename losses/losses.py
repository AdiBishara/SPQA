import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

def get_3d_active_contour_length(probs):
    # Added epsilon inside sqrt for stability
    dx = torch.pow(probs[:, :, 1:, :-1, :-1] - probs[:, :, :-1, :-1, :-1], 2)
    dy = torch.pow(probs[:, :, :-1, 1:, :-1] - probs[:, :, :-1, :-1, :-1], 2)
    dz = torch.pow(probs[:, :, :-1, :-1, 1:] - probs[:, :, :-1, :-1, :-1], 2)
    return torch.sum(torch.sqrt(dx + dy + dz + 1e-8))

def calc_dist_map_batch(y_true):
    y_true_np = y_true.detach().cpu().numpy().astype(bool)
    dist_map = np.zeros_like(y_true_np).astype(np.float32)
    for b in range(y_true_np.shape[0]):
        for c in range(y_true_np.shape[1]):
            posmask = y_true_np[b, c]
            if posmask.any():
                negmask = ~posmask
                dist_map[b, c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return torch.from_numpy(dist_map).to(y_true.device)

def dice_coefficient(probs, target, smooth=1.0):
    # Changed smooth to 1.0 for better stability
    intersection = (probs * target).sum()
    return (2. * intersection + smooth) / (probs.sum() + target.sum() + smooth)

class VAELoss(nn.Module):
    def __init__(self, config, kld_weight=0.005):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight
        self.w = config['LossWeights']
        # Helper for standard BCE
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, recon_x, x, mu, logvar, corrupted_input=None, beta=None, calculate_boundary=False):
        # Apply sigmoid once
        probs = torch.sigmoid(recon_x)

        # 1. Regional Dice
        d_coef = dice_coefficient(probs, x)
        d_loss = 1.0 - d_coef

        # 2. Boundary Loss (SDM)
        boundary_l = torch.tensor(0.0, device=x.device)
        if calculate_boundary:
            with torch.no_grad():
                dist_map = calc_dist_map_batch(x)
            # Multiplying probs by distance map pushes boundaries inward/outward
            boundary_l = torch.mean(probs * dist_map)

        # 3. Active Contour Length (SAFEGUARDED)
        len_pred = get_3d_active_contour_length(probs)
        len_true = get_3d_active_contour_length(x)
        # Clamp the ratio so noise doesn't explode the gradient
        length_ratio = torch.clamp(len_pred / (len_true + 1e-8), max=5.0)

        # 4. Standard Pixel-wise BCE (Crucial for stability)
        pixel_loss = self.bce(recon_x, x)

        # 5. Composite Recon Loss
        # We use 'bce' from config for the SDM weight (as per your setup)
        # We add 'pixel_loss' (weighted by 1.0) to ensure convergence
        recon_total = (self.w['dice'] * d_loss) + \
                      (self.w['bce'] * boundary_l) + \
                      (self.w['laplace'] * length_ratio) + \
                      pixel_loss

        # 6. Fixation Loss (if using corrupted input)
        if corrupted_input is not None:
            error_mask = torch.abs(corrupted_input - x)
            # Only calculate fix loss if there is actually an error to fix
            if error_mask.sum() > 0:
                fix_l = 1.0 - dice_coefficient(probs * error_mask, x * error_mask)
                recon_total += (self.w['fix_weight'] * fix_l)

        # 7. KLD
        kld_l = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        k_weight = beta if beta is not None else self.kld_weight

        total = recon_total + (k_weight * (kld_l / mu.shape[1]))

        return total, d_coef, boundary_l, length_ratio