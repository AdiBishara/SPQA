import os
import sys
import glob
import re
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Project Imports
from utils.config import load_config
from utils.seeding import fix_seeds
from utils.data.nifti_loader import NiftiDataset
from utils.models.vae import VAE3D


# --- DYNAMIC MORPHOLOGICAL CORRUPTION ---
def morphological_corruption(mask, kernel_size=3, max_iters=5):
    """Randomly dilates or erodes the mask to create training samples for the QC VAE."""
    mode = "dilate" if torch.rand(1).item() > 0.5 else "erode"
    iters = torch.randint(1, max_iters + 1, (1,)).item()
    pad = kernel_size // 2
    corrupted = mask.clone()
    if mode == "dilate":
        for _ in range(iters):
            corrupted = F.max_pool3d(corrupted, kernel_size=kernel_size, stride=1, padding=pad)
    else:
        corrupted = -corrupted
        for _ in range(iters):
            corrupted = F.max_pool3d(corrupted, kernel_size=kernel_size, stride=1, padding=pad)
        corrupted = -corrupted
    return corrupted


# --- LOSS HELPERS ---
def robust_dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def get_laplacian_loss(recon_logits, target):
    """
    Penalizes blurriness by comparing the edge-maps of the reconstruction and the truth.
    Forces the model to learn fine brain contours (gyri/sulci).
    """
    # 3D Laplacian kernel: [1, 1, 3, 3, 3]
    kernel = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                          dtype=torch.float32, device=recon_logits.device).view(1, 1, 3, 3, 3)

    recon_probs = torch.sigmoid(recon_logits)
    edge_recon = F.conv3d(recon_probs, kernel, padding=1)
    edge_target = F.conv3d(target, kernel, padding=1)

    return F.mse_loss(edge_recon, edge_target)


# --- FINAL HIGH-FIDELITY LOSS FUNCTION ---
def vae_loss_function(recon, target, mu, logvar, beta=0.0):
    """
    Phase 2 Loss: Priority shifted to edge sharpness and anatomical detail.
    """
    # 1. Weighted BCE: 10x penalty for errors on brain pixels (foreground)
    pos_weight = torch.tensor([10.0], device=recon.device)
    bce_loss = F.binary_cross_entropy_with_logits(recon, target, pos_weight=pos_weight)

    # 2. Laplacian Edge Loss: Specifically targets high-frequency contour details
    edge_loss = get_laplacian_loss(recon, target)

    # 3. Dice Loss: Global shape topology
    recon_probs = torch.sigmoid(recon)
    dice_loss = robust_dice_loss(recon_probs, target).mean()

    # 4. KL Divergence: Regularization (Normalized)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss /= (target.numel())

    # BALANCE: 50% BCE (Location), 40% Edge (Contours), 10% Dice (Overlap)
    total_loss = (0.5 * bce_loss) + (0.4 * edge_loss) + (0.1 * dice_loss) + (beta * kld_loss)

    return total_loss, dice_loss, edge_loss, kld_loss


# --- LOGGER ---
class HybridLogger(object):
    def __init__(self, filepath, resume=False):
        mode = "a" if resume else "w"
        self.log = open(filepath, mode, encoding='utf-8')
        self.terminal = sys.stdout

    def write(self, message):
        self.log.write(message)
        self.log.flush()
        # Clean terminal output
        clean_keywords = ["Epoch", "Total", "Edge", "Saved", "Starting"]
        if any(k in message for k in clean_keywords) or message.strip() == "":
            self.terminal.write(message)
            self.terminal.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_next_log_file(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_logs = glob.glob(os.path.join(base_dir, "vae_run_*.txt"))
    max_run = 0
    for log_file in existing_logs:
        try:
            r = int(os.path.basename(log_file).split('_')[-1].split('.')[0])
            if r > max_run: max_run = r
        except:
            continue
    return os.path.join(base_dir, f"vae_run_{max_run + 1}.txt")


def train_vae():
    # 1. INITIAL SETUP
    config_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml"
    config = load_config(config_path)
    fix_seeds(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\training_logs"
    log_file = get_next_log_file(log_dir)
    sys.stdout = HybridLogger(log_file)

    print(f"--- STARTING RESIDUAL HIGH-FIDELITY VAE TRAINING ---")
    print("✅ Architecture: Residual with 8x8x8 Bottleneck")
    print("✅ Sharpener: Laplacian Edge Loss Integrated")

    save_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\vae_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 2. DATA LOADING
    dataset = NiftiDataset(
        img_dir=config['Data']['raw_data_root'],
        list_path=config['Data']['training_ids'],
        image_size=config['model']['image_size'],
        is_train=True
    )

    loader = DataLoader(
        dataset,
        batch_size=config['Train']['batch_size'],
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )

    # 3. MODEL & OPTIMIZER
    vae = VAE3D(
        in_channels=2,  # Image + Mask
        out_channels=1,
        image_size=config['model']['image_size'],
        latent_dim=config['model']['latent_dim']  # 2048 recommended
    ).to(device)

    # Low Learning Rate for detail stability
    optimizer = optim.Adam(vae.parameters(), lr=5e-5)

    # 4. TRAINING LOOP
    vae.train()
    print("Training initialized.")

    for epoch in range(config['Train']['epochs']):
        epoch_total, epoch_dice, epoch_edge, epoch_kld = 0, 0, 0, 0
        valid_batches = 0

        # Beta schedule (Small to allow unconstrained reconstruction)
        if epoch < 50:
            beta = 1e-7
        else:
            beta = 0.0005 * ((epoch - 50) / 100)
            if beta > 0.0005: beta = 0.0005

        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            clean_masks = batch['mask'].to(device, non_blocking=True)

            # Corrupt the mask for the denoising task
            corrupted_masks = morphological_corruption(clean_masks, kernel_size=3, max_iters=5)
            vae_input = torch.cat([images, corrupted_masks], dim=1)

            optimizer.zero_grad()
            recon, mu, logvar = vae(vae_input)

            loss, d_loss, e_loss, k_loss = vae_loss_function(recon, clean_masks, mu, logvar, beta=beta)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_total += loss.item()
            epoch_dice += d_loss.item()
            epoch_edge += e_loss.item()
            epoch_kld += k_loss.item()
            valid_batches += 1

        div = valid_batches if valid_batches > 0 else 1
        print(
            f"Epoch {epoch + 1:03d} | Total: {epoch_total / div:.4f} | DiceLoss: {epoch_dice / div:.4f} | EdgeLoss: {epoch_edge / div:.6f} | KLD: {epoch_kld / div:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(vae.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch + 1}.pth"))
            print(f"   -> Saved Checkpoint (Epoch {epoch + 1})")


if __name__ == "__main__":
    train_vae()