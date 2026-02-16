import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
import re
import glob

# --- FIX: FORCE HEADLESS BACKEND ---
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.seeding import fix_seeds
from utils.data.nifti_loader import NiftiDataset
from utils.models.vae import VAE3D
from losses.losses import VAELoss


# --- 0. UTILS: FIND LATEST CHECKPOINT ---
def find_latest_checkpoint(save_dir):
    checkpoints = glob.glob(os.path.join(save_dir, "vae_epoch_*.pth"))
    if not checkpoints:
        return None, 0

    def extract_epoch(ckpt_path):
        match = re.search(r'vae_epoch_(\d+).pth', ckpt_path)
        return int(match.group(1)) if match else 0

    latest_ckpt = max(checkpoints, key=extract_epoch)
    latest_epoch = extract_epoch(latest_ckpt)
    return latest_ckpt, latest_epoch


# --- 1. SEQUENTIAL LOGGER ---
class SequentialLogger(object):
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        existing_runs = [f for f in os.listdir(log_dir) if f.startswith("vae_run_")]
        run_numbers = [int(f.split("_")[-1].replace(".txt", "")) for f in existing_runs if
                       f.split("_")[-1].replace(".txt", "").isdigit()]
        self.run_number = max(run_numbers) + 1 if run_numbers else 1
        self.log_name = f"vae_run_{self.run_number}"
        self.log_path = os.path.join(log_dir, f"{self.log_name}.txt")
        self.terminal = sys.stdout
        self.log = open(self.log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def write_file_only(self, message):
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# --- 2. HIGH-CONTRAST VISUALIZER ---
def save_visual_check(recon, target, image, epoch, save_dir):
    try:
        os.makedirs(save_dir, exist_ok=True)
        d, h, w = image.shape[2:]
        mid_d = d // 2
        slices = [mid_d, min(mid_d + 15, d - 1), max(mid_d - 15, 0)]
        fig, axes = plt.subplots(len(slices), 3, figsize=(15, 5 * len(slices)))
        for idx, s_idx in enumerate(slices):
            img_slice = image[0, 0, s_idx].detach().cpu().numpy()
            gt_slice = target[0, 0, s_idx].detach().cpu().numpy()
            probs = torch.sigmoid(recon[0, 0, s_idx]).detach().cpu().numpy()
            pred_slice = (probs > 0.5).astype(np.float32)
            axes[idx, 0].imshow(img_slice, cmap='gray')
            axes[idx, 0].set_title(f"MRI Slice {s_idx}")
            axes[idx, 1].imshow(img_slice, cmap='gray')
            axes[idx, 1].imshow(gt_slice, cmap='Greens', alpha=0.4)
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 2].imshow(img_slice, cmap='gray')
            axes[idx, 2].imshow(pred_slice, cmap='Reds', alpha=0.4)
            axes[idx, 2].set_title(f"VAE Pred (Epoch {epoch})")
            for ax in axes[idx]: ax.axis('off')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_check.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        return save_path
    except Exception as e:
        print(f"   [VISUAL ERROR]: {e}")
        return None


# --- 3. DYNAMIC SCHEDULER (AGGRESSIVE EDGE HUNTER) ---
def update_weights_damped(criterion, avg_dice, stable_counter, epoch):
    current_fix = criterion.w.get('fix_weight', 1.0)
    new_weights = criterion.w.copy()

    # FORCED PROGRESSION: Don't get stuck in Phase 2
    # Once we have high enough dice, we switch to extreme boundary focus

    # PHASE 4: ANATOMY LOCK (Zero Smoothing, Max Boundary)
    if avg_dice >= 0.90 and current_fix >= 6.0:
        new_weights.update({'dice': 20.0, 'bce': 0.15, 'laplace': 0.0, 'fix_weight': 10.0, 'boundary': 5.0})
        phase = "Phase 4 (Anatomy Lock - MAX BOUNDARY)"

    # PHASE 3: PRECISION (Minimal Smoothing, High Boundary)
    elif (avg_dice >= 0.82 or epoch > 200) and current_fix >= 3.0:
        new_weights.update({'dice': 20.0, 'bce': 0.10, 'laplace': 0.001, 'fix_weight': 6.0, 'boundary': 3.0})
        phase = "Phase 3 (Precision - EDGE HUNTER)"

    # PHASE 2: SHARPENING (Moderate Smoothing)
    elif avg_dice >= 0.70:
        new_weights.update({'dice': 20.0, 'bce': 0.05, 'laplace': 0.05, 'fix_weight': 3.0, 'boundary': 0.8})
        phase = "Phase 2 (Sharpening)"

    # PHASE 1: VOLUME (Default)
    else:
        new_weights.update({'dice': 10.0, 'bce': 0.0, 'laplace': 0.0, 'fix_weight': 1.0, 'boundary': 0.0})
        phase = "Phase 1 (Volume)"

    criterion.w = new_weights
    return phase


# --- 4. CORRUPTION ---
def morphological_corruption(mask):
    with torch.no_grad():
        mode = "dilate" if torch.rand(1).item() > 0.5 else "erode"
        iters = torch.randint(1, 6, (1,)).item()
        corrupted = mask.clone()
        for _ in range(iters):
            if mode == "dilate":
                corrupted = F.max_pool3d(corrupted, kernel_size=3, stride=1, padding=1)
            else:
                corrupted = -F.max_pool3d(-corrupted, kernel_size=3, stride=1, padding=1)
    return corrupted


# --- 5. TRAINING LOOP ---
def train_vae():
    config = load_config(r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml")
    fix_seeds(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\training_logs"
    save_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\vae_checkpoints"
    vis_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\visual_progress"
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    logger = SequentialLogger(log_dir)
    sys.stdout = logger

    model = VAE3D(in_channels=2, out_channels=1, latent_dim=2048).to(device)
    ckpt_path, start_epoch = find_latest_checkpoint(save_dir)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = VAELoss(config=config, kld_weight=0.005).to(device)
    bce_stable = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    scaler = GradScaler('cuda')

    if ckpt_path:
        print(f"RESUMING FROM EPOCH: {start_epoch}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("STARTING NEW TRAINING")

    dataset = NiftiDataset(img_dir=config['Data']['raw_data_root'], list_path=config['Data']['training_ids'],
                           image_size=config['model']['image_size'], is_train=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    rolling_dice = 0.0
    stable_counter = 0

    for epoch in range(start_epoch, config['Train']['epochs']):
        model.train()
        epoch_dice, epoch_loss, count = 0, 0, 0
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            img, mask = batch['image'].to(device), (batch['mask'].to(device) > 0.5).float()
            corr = morphological_corruption(mask)
            with autocast('cuda'):
                recon, mu, logvar = model(torch.cat([img, corr], dim=1))
                d_acc = (2. * (torch.sigmoid(recon) * mask).sum()) / (torch.sigmoid(recon).sum() + mask.sum() + 1e-6)
                if rolling_dice < 0.70:
                    loss = bce_stable(recon, mask) + (1.0 - d_acc)
                else:
                    loss, _, _, _ = criterion(recon, mask, mu, logvar, corrupted_input=corr, calculate_boundary=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_dice += d_acc.item()
            epoch_loss += loss.item()
            count += 1
            if i == 0 and (epoch == start_epoch or (epoch + 1) % 10 == 0):
                v_path = save_visual_check(recon, mask, img, epoch + 1, vis_dir)
                if v_path: print(f"   [VISUAL SAVED]: {os.path.basename(v_path)}")

        new_avg_dice = epoch_dice / count
        stable_counter = stable_counter + 1 if new_avg_dice >= rolling_dice - 0.03 else 0  # More forgiving
        rolling_dice = new_avg_dice

        # Call with epoch to enable forced progression
        current_phase = update_weights_damped(criterion, rolling_dice, stable_counter, epoch + 1)

        print(f"Epoch {epoch + 1:03d} | {current_phase} | Dice: {rolling_dice:.4f} | Loss: {epoch_loss / count:.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch + 1}.pth"))


if __name__ == "__main__":
    train_vae()