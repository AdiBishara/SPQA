import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Standard Segmentation Losses ---
bce_criterion = nn.BCEWithLogitsLoss()


def dice_loss(pred, target, smooth=1e-6):
    """
    Standard Dice Loss.
    Pred: (B, C, H, W) or (B, C, D, H, W) -> Logits or Probabilities
    Target: (B, C, H, W) or (B, C, D, H, W) -> Binary Mask (0 or 1)
    """
    # Ensure pred is sigmoid-activated if it comes as raw logits
    # But usually we handle that outside or assume inputs are effectively probabilities for the calculation

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


# --- 2. THE MISSING PIECE: VAE LOSS ---
class VAELoss(nn.Module):
    def __init__(self, kld_weight=0.005):  # Increased weight to punish "Skull Copying"
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, recon_x, x, mu, logvar):
        """
        recon_x: VAE Output (Logits)
        x:       Target Mask (Ground Truth)
        mu:      Latent Mean
        logvar:  Latent Log-Variance
        """
        # 1. Reconstruction Loss (Dice + BCE)
        # We combine Dice (Structure) and BCE (Pixel-wise accuracy) for stability
        pred_prob = torch.sigmoid(recon_x)
        d_loss = dice_loss(pred_prob, x)
        bce_loss = F.binary_cross_entropy_with_logits(recon_x, x)

        # Weighted Recon Loss (mostly Dice)
        recon_loss = d_loss + (0.5 * bce_loss)

        # 2. KLD Loss (The "Regularizer")
        # Forces the model to learn a smooth "Concept" of a brain, not just memorize pixels.
        # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

        # 3. Total Loss
        total_loss = recon_loss + (self.kld_weight * kld_loss)

        return total_loss, recon_loss, kld_loss