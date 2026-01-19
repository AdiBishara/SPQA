import os
import sys
import glob
import re
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Project Imports
from utils.config import load_config
from utils.seeding import fix_seeds
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.models.vae import VAE3D


# --- DISABLE AUGMENTATION FOR STABILITY ---
# We remove the GPU Augmentation for the VAE.
# It needs to learn to reconstruct stable images first.

# --- ROBUST LOSS ---
def robust_dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def vae_loss_function(recon, target, mu, logvar, beta=0.0):
    # 1. Reconstruction Loss (Dice)
    # Add epsilon to sigmoid to prevent pure 0 or 1
    recon_probs = torch.sigmoid(recon)
    recon_probs = torch.clamp(recon_probs, 1e-6, 1 - 1e-6)
    recon_loss = robust_dice_loss(recon_probs, target).mean()

    # 2. Safety Clamp for Latent Space (The Stabilizer)
    # Tighter clamp (-5 to 5) prevents massive exponentiations
    logvar = torch.clamp(logvar, min=-5, max=5)
    mu = torch.clamp(mu, min=-10, max=10)

    # 3. KL Divergence
    num_voxels = target.numel()
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss /= num_voxels

    return recon_loss + (beta * kld_loss), recon_loss, kld_loss


# --- FIXED LOGGER ---
class HybridLogger(object):
    def __init__(self, filepath, resume=False):
        mode = "a" if resume else "w"
        self.log = open(filepath, mode, encoding='utf-8')
        self.terminal = sys.stdout

    def write(self, message):
        self.log.write(message)
        self.log.flush()
        # ALLOW WARNINGS THROUGH
        clean_keywords = ["Epoch", "Total", "Loading", "Starting", "Saved", "WARNING", "NaN"]
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
    # 1. SETUP
    config_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml"
    config = load_config(config_path)
    fix_seeds(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\training_logs"
    log_file = get_next_log_file(log_dir)
    sys.stdout = HybridLogger(log_file)

    print(f"--- Starting VAE Training on {device} ---")

    save_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\vae_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 2. DATA LOADER
    dataset = NiftiFewShotDataset(
        data_root=config['Data']['raw_data_root'],
        id_file=config['Data']['training_ids'],
        image_size=config['model']['image_size'],
        is_train=True
    )
    batch_size = config['Train']['batch_size']

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )

    # 3. MODEL
    vae = VAE3D(
        in_channels=2,
        out_channels=1,
        image_size=config['model']['image_size'],
        latent_dim=config['model']['latent_dim']
    ).to(device)

    # LOWER LEARNING RATE (Critical for stability)
    optimizer = optim.Adam(vae.parameters(), lr=2e-5)

    # 4. LOOP
    vae.train()

    # Reset to epoch 0 for stability (or you can load weights if you trust them)
    print("Training from scratch to ensure stability.")

    for epoch in range(config['Train']['epochs']):
        epoch_total = 0
        epoch_recon = 0
        epoch_kld = 0
        valid_batches = 0

        # Slower Annealing: 0 -> 0.005 over 100 epochs
        beta = 0.005 * (epoch / 100) if epoch < 100 else 0.005

        for images, clean_masks, _ in loader:
            images, clean_masks = images.to(device, non_blocking=True), clean_masks.to(device, non_blocking=True)

            # --- CUTOUT CORRUPTION ---
            corrupted_masks = clean_masks.clone()
            B, C, D, H, W = corrupted_masks.shape

            for b in range(B):
                d_box = torch.randint(D // 10, D // 5, (1,)).item()
                h_box = torch.randint(H // 10, H // 5, (1,)).item()
                w_box = torch.randint(W // 10, W // 5, (1,)).item()

                z = torch.randint(0, D - d_box, (1,)).item()
                y = torch.randint(0, H - h_box, (1,)).item()
                x = torch.randint(0, W - w_box, (1,)).item()

                val = 0.0 if torch.rand(1).item() > 0.5 else 1.0
                corrupted_masks[b, :, z:z + d_box, y:y + h_box, x:x + w_box] = val
            # -------------------------

            vae_input = torch.cat([images, corrupted_masks], dim=1)

            optimizer.zero_grad()

            # PURE FP32 (No Autocast)
            recon, mu, logvar = vae(vae_input)
            loss, r_loss, k_loss = vae_loss_function(recon, clean_masks, mu, logvar, beta=beta)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ NaN Detected at Epoch {epoch + 1}! Skipping batch.")
                continue

            loss.backward()
            # Stricter Gradient Clipping
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += r_loss.item()
            epoch_kld += k_loss.item()
            valid_batches += 1

        div = valid_batches if valid_batches > 0 else 1

        print(
            f"Epoch {epoch + 1} | Beta: {beta:.4f} | Total: {epoch_total / div:.4f} | Recon: {epoch_recon / div:.4f} | KLD: {epoch_kld / div:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(vae.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch + 1}.pth"))
            print(f"   -> Saved Checkpoint (Epoch {epoch + 1})")


if __name__ == "__main__":
    train_vae()