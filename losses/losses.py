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

        # Standard BCE used for intensity matching
        # Reduction 'none' so we can apply the band mask manually
        # PyTorch Autocast REQUIRES BCEWithLogitsLoss for float16 stability!
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, recon_x, x, mu, logvar):
        # x is the Ground Truth target
        # recon_x contains RAW LOGITS. 
        # We compute probs here strictly for Dice/Contour calculations:
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

        # 4. Contour Loss (SDM boundary overlap)
        contour_l = torch.tensor(0.0, device=x.device)
        if self.w.get('contour', 0.0) > 0:
            contour_l = torch.mean(probs * dist_map)

        # 5. Composite Reconstruction Loss
        recon_total = (self.w.get('bce', 0.125) * bce_loss) + \
                      (self.w.get('dice', 8.0) * d_loss) + \
                      (self.w.get('contour', 0.0) * contour_l)

        # 6. Optional KLD (if kld_weight > 0.0, acts as Variational DAE)
        kld_l = torch.tensor(0.0, device=x.device)
        if self.kld_weight > 0.0 and mu is not None and logvar is not None:
            kld_l = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            total = recon_total + (self.kld_weight * (kld_l / mu.shape[1]))
        else:
            total = recon_total

        # --- MATCHING THE TRAINER RETURN SIGNATURE ---
        # Returns Total, Dice Acc, Contour Loss, BCE Loss, KLD Loss
        return total, d_acc, contour_l, bce_loss, kld_l
