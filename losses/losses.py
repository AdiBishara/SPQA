import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

# --- HELPER FUNCTIONS ---
def get_3d_active_contour_length(probs):
    """Calculates 3D active contour length for boundary regularization."""
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

def dice_loss(probs, target, smooth=1.0):
    """Standard Dice Loss computation (1 - Dice)."""
    return 1.0 - dice_coefficient(probs, target, smooth)

# --- CORE LOSS CLASS ---
class DAELoss(nn.Module):
    def __init__(self, config):
        super(DAELoss, self).__init__()
        
        # Load KLD weight (can be >0 for probabilistic DAE)
        self.kld_weight = config.get('train', {}).get('kld_weight', 0.0)

        # Synchronized with the dynamic config-driven loader
        initial_phase = config['phases']['phase1_volume']
        self.w = {k: v for k, v in initial_phase.items() if k != 'threshold'}

        # BCE with 'none' reduction allows manual band masking.
        # BCEWithLogitsLoss is required for PyTorch Autocast float16 stability.
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, recon_x, x, mu, logvar):
        # x is Ground Truth; recon_x contains raw logits.
        # Compute probs strictly for Dice/Contour metrics:
        probs = torch.sigmoid(recon_x)

        # 1. SDM calculation (needed for Contour and Band Size)
        with torch.no_grad():
            dist_map = calc_dist_map_batch(x)

        # 2. BCE Loss (with Dynamic Band Size applied)
        band_size = self.w.get('band_size', 0)
        bce_raw = self.bce(recon_x, x)
        if band_size > 0:
            band_mask = (torch.abs(dist_map) <= band_size).float()
            if band_mask.sum() > 0:
                bce_loss = (bce_raw * band_mask).sum() / band_mask.sum()
            else:
                bce_loss = bce_raw.mean()
        else:
            bce_loss = bce_raw.mean()

        # 3. Regional Dice Loss
        d_acc = dice_coefficient(probs, x)
        d_loss = dice_loss(probs, x)

        # 4. Contour Loss (Boundary Confidence)
        # Regional Dice localized to the boundary to explicitly reward precise edge matching.
        contour_l = torch.tensor(0.0, device=x.device)
        if self.w.get('contour', 0.0) > 0:
            # Isolate the literal 3D boundary of the Ground Truth mask (where SDM is between -2 and 2)
            boundary_mask = (torch.abs(dist_map) <= 2.0).float()
            
            if boundary_mask.sum() > 0:
                b_probs = probs * boundary_mask
                b_target = x * boundary_mask
                
                # Localized Boundary Dice Loss
                intersection = (b_probs * b_target).sum()
                contour_l = 1.0 - (2. * intersection + 1.0) / (b_probs.sum() + b_target.sum() + 1.0)

        # 5. Composite Reconstruction Loss
        recon_total = (self.w.get('bce', 0.125) * bce_loss) + \
                      (self.w.get('dice', 8.0) * d_loss) + \
                      (self.w.get('contour', 0.0) * contour_l)

        # 6. Optional KLD (if kld > 0.0, acts as Variational DAE)
        kld_l = torch.tensor(0.0, device=x.device)
        current_kld_weight = self.w.get('kld', self.kld_weight)
        
        if current_kld_weight > 0.0 and mu is not None and logvar is not None:
            # Force float32 to prevent float16 KLD exponentiation overflow
            mu_f32 = mu.to(torch.float32)
            logvar_f32 = logvar.to(torch.float32)
            
            # Flatten spatial dims to dynamically support any bottleneck shape (1D/2D/3D)
            mu_flat = mu_f32.view(mu_f32.size(0), -1)
            logvar_flat = logvar_f32.view(logvar_f32.size(0), -1)
            
            # KLD = -0.5 * sum(1 + log(var) - mu^2 - var); sum over latents, average over batch
            kld_raw = -0.5 * torch.sum(1 + logvar_flat - mu_flat.pow(2) - logvar_flat.exp(), dim=1).mean()
            kld_l = kld_raw.to(x.dtype)
            
            total = recon_total + (current_kld_weight * (kld_l / mu_flat.shape[1]))
        else:
            total = recon_total

        # Returns: Total Loss, Dice Acc, Contour Loss, BCE Loss, KLD Loss
        return total, d_acc, contour_l, bce_loss, kld_l
