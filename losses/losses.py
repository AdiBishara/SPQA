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

        # FIXED: Initialize using the tidied lowercase 'phases' structure
        # We start with the phase1_volume weights. These are updated by the trainer later.
        initial_phase = config['phases']['phase1_volume']
        self.w = {k: v for k, v in initial_phase.items() if k != 'threshold'}

        # Helper for standard BCE (Pixel-wise stability)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, recon_x, x, mu, logvar, corrupted_input=None, beta=None, calculate_boundary=False):
        # Apply sigmoid once for all probability-based calculations
        probs = torch.sigmoid(recon_x)

        # 1. Regional Dice Loss (Volume focus)
        d_coef = dice_coefficient(probs, x)
        d_loss = 1.0 - d_coef

        # 2. Boundary Loss (SDM) - Critical for SPQA quality flagging
        boundary_l = torch.tensor(0.0, device=x.device)
        if calculate_boundary:
            with torch.no_grad():
                dist_map = calc_dist_map_batch(x)
            # Multiplying probabilities by the distance map penalizes boundary errors heavily
            boundary_l = torch.mean(probs * dist_map)

        # 3. Active Contour Length (Safeguarded)
        len_pred = get_3d_active_contour_length(probs)
        len_true = get_3d_active_contour_length(x)
        # Clamp the ratio to prevent noise from exploding the gradients
        length_ratio = torch.clamp(len_pred / (len_true + 1e-8), max=5.0)

        # 4. Standard Pixel-wise BCE (Ensures basic convergence)
        pixel_loss = self.bce(recon_x, x)

        # 5. Composite Reconstruction Loss
        # Dynamically uses weights updated by the trainer's Phase logic
        recon_total = (self.w['dice'] * d_loss) + \
                      (self.w['boundary'] * boundary_l) + \
                      (self.w['laplace'] * length_ratio) + \
                      pixel_loss

        # 6. Fixation Loss (Active during VAE corruption training)
        if corrupted_input is not None:
            error_mask = torch.abs(corrupted_input - x)
            if error_mask.sum() > 0:
                # Forces the model to focus specifically on fixing the added artifacts
                fix_l = 1.0 - dice_coefficient(probs * error_mask, x * error_mask)
                recon_total += (self.w['fix_weight'] * fix_l)

        # 7. Kullback-Leibler Divergence (KLD)
        kld_l = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        k_weight = beta if beta is not None else self.kld_weight

        # Final Total Loss
        total = recon_total + (k_weight * (kld_l / mu.shape[1]))

        return total, d_coef, boundary_l, length_ratio