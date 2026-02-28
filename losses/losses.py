import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

# --- HELPER FUNCTIONS ---
def get_3d_active_contour_length(probs):
    """Calculates the length of the active contour in 3D for boundary regularization."""
    # Added epsilon inside sqrt for stability
    dx = torch.pow(probs[:, :, 1:, :-1, :-1] - probs[:, :, :-1, :-1, :-1], 2)
    dy = torch.pow(probs[:, :, :-1, 1:, :-1] - probs[:, :, :-1, :-1, :-1], 2)
    dz = torch.pow(probs[:, :, :-1, :-1, 1:] - probs[:, :, :-1, :-1, :-1], 2)
    return torch.sum(torch.sqrt(dx + dy + dz + 1e-8))

def calc_dist_map_batch(y_true):
    """Calculates the Signed Distance Map (SDM) for boundary loss."""
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
    """Calculates the regional Dice coefficient for mask overlap."""
    intersection = (probs * target).sum()
    return (2. * intersection + smooth) / (probs.sum() + target.sum() + smooth)

# --- CORE LOSS CLASS ---
class VAELoss(nn.Module):
    def __init__(self, config, kld_weight=0.005):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight

        # Synchronized with the dynamic config-driven loader
        initial_phase = config['phases']['phase1_volume']
        self.w = {k: v for k, v in initial_phase.items() if k != 'threshold'}

        # Standard BCE used for intensity matching (Pixel Loss)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, recon_x, x, mu, logvar, corrupted_input=None, beta=None):
        # 1. Probabilities for all Dice and Boundary calculations
        probs = torch.sigmoid(recon_x)

        # 2. Regional Dice Loss (Volume focus)
        d_acc = dice_coefficient(probs, x)
        d_loss = 1.0 - d_acc

        # 3. Boundary Loss (SDM) - Auto-enabled when boundary weight > 0
        boundary_l = torch.tensor(0.0, device=x.device)
        if self.w.get('boundary', 0.0) > 0:
            with torch.no_grad():
                dist_map = calc_dist_map_batch(x)
            boundary_l = torch.mean(probs * dist_map)

        # 4. Active Contour Length (Laplace Ratio)
        len_pred = get_3d_active_contour_length(probs)
        len_true = get_3d_active_contour_length(x)
        length_ratio = torch.clamp(len_pred / (len_true + 1e-8), max=5.0)

        # 5. Pixel-wise BCE Loss
        pixel_loss = self.bce(recon_x, x)

        # 6. Fixation Loss (Active during discovery and corruption training)
        fix_l = torch.tensor(0.0, device=x.device)
        if corrupted_input is not None:
            error_mask = torch.abs(corrupted_input - x)
            if error_mask.sum() > 0:
                # Forces model to focus specifically on fixing the added artifacts
                fix_l = 1.0 - dice_coefficient(probs * error_mask, x * error_mask)

        # 7. Composite Reconstruction Loss using weights from config
        recon_total = (self.w.get('dice', 8.0) * d_loss) + \
                      (self.w.get('boundary', 0.0) * boundary_l) + \
                      (self.w.get('laplace', 0.0) * length_ratio) + \
                      (self.w.get('pixel', 0.125) * pixel_loss) + \
                      (self.w.get('fix_weight', 0.125) * fix_l)

        # 8. Kullback-Leibler Divergence (KLD)
        kld_l = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        k_weight = beta if beta is not None else self.kld_weight

        # Final Total Loss
        total = recon_total + (k_weight * (kld_l / mu.shape[1]))

        # --- MATCHING THE TRAINER RETURN SIGNATURE ---
        # (Total, Dice_Acc, Boundary, Laplace, Pixel, Fixation)
        return total, d_acc, boundary_l, length_ratio, pixel_loss, fix_l