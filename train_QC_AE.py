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
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Custom imports
from utils.config import load_config
from utils.seeding import fix_seeds
from utils.data.nifti_loader import NiftiDataset
from utils.models.vae import VAE3D
from losses.losses import VAELoss


# --- 0. DYNAMIC WEIGHT SCHEDULER ---
def update_weights_from_config(criterion, avg_dice, config):
    """Finds the highest qualified phase from config.yaml based on avg_dice."""
    phases = config['phases']
    qualified_phases = [
        (name, data) for name, data in phases.items()
        if avg_dice >= data['threshold']
    ]
    # Pick the one with the highest threshold
    best_phase_name, best_phase_data = max(qualified_phases, key=lambda x: x[1]['threshold'])

    # Update criterion weights, removing the threshold key
    new_weights = {k: v for k, v in best_phase_data.items() if k != 'threshold'}
    criterion.w = new_weights

    return best_phase_name


# --- 1. ENHANCED UTILS & LOGGING ---
def find_latest_checkpoint(save_dir):
    """Finds the latest vae_epoch_*.pth file."""
    checkpoints = glob.glob(os.path.join(save_dir, "vae_epoch_*.pth"))
    if not checkpoints: return None, 0

    def extract_epoch(ckpt_path):
        match = re.search(r'vae_epoch_(\d+).pth', ckpt_path)
        return int(match.group(1)) if match else 0

    latest_ckpt = max(checkpoints, key=extract_epoch)
    return latest_ckpt, extract_epoch(latest_ckpt)


class SequentialLogger(object):
    """Logs to console and a sequentially numbered txt file using UTF-8."""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        existing_logs = glob.glob(os.path.join(log_dir, "vae_run_*.txt"))
        run_numbers = [int(re.search(r'vae_run_(\d+).txt', f).group(1)) for f in existing_logs if
                       re.search(r'vae_run_(\d+).txt', f)]
        self.run_number = max(run_numbers) + 1 if run_numbers else 1

        self.log_path = os.path.join(log_dir, f"vae_run_{self.run_number}.txt")
        self.terminal = sys.stdout

        # Using UTF-8 to ensure all characters are handled safely
        self.log = open(self.log_path, "a", encoding="utf-8")
        print(f"LOGGING SESSION {self.run_number} TO: {self.log_path}")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def save_visual_check(recon, target, image, epoch, save_dir, subject_id):
    """Saves 3 anatomical slices with Subject ID labeling."""
    os.makedirs(save_dir, exist_ok=True)
    d, h, w = image.shape[2:]

    # Axial slices at 25%, 50%, and 75% depth
    slices = {
        "Inferior": int(d * 0.25),
        "Central": int(d * 0.5),
        "Superior": int(d * 0.75)
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f"VAE Visual Validation | Epoch {epoch} | Subject: {subject_id}", fontsize=20, fontweight='bold')

    for idx, (label, s_idx) in enumerate(slices.items()):
        img_slice = image[0, 0, s_idx].detach().cpu().numpy()
        gt_slice = target[0, 0, s_idx].detach().cpu().numpy()
        probs = torch.sigmoid(recon[0, 0, s_idx]).detach().cpu().numpy()
        pred_slice = (probs > 0.5).astype(np.float32)

        # MRI Background
        axes[idx, 0].imshow(img_slice, cmap='gray')
        axes[idx, 0].set_title(f"{label} Slice ({s_idx}) - MRI Scan")

        # GT Overlay
        axes[idx, 1].imshow(img_slice, cmap='gray')
        axes[idx, 1].imshow(gt_slice, cmap='Greens', alpha=0.4)
        axes[idx, 1].set_title("Ground Truth")

        # VAE Overlay
        axes[idx, 2].imshow(img_slice, cmap='gray')
        axes[idx, 2].imshow(pred_slice, cmap='Reds', alpha=0.4)
        axes[idx, 2].set_title("VAE Prediction")

        for ax in axes[idx]:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_{subject_id}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


def morphological_corruption(mask):
    """Applies random erosion or dilation to simulate artifacts."""
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


# --- 2. TRAINING LOOP ---
def train_vae():
    config = load_config(r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml")
    fix_seeds(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up Logger - All prints mirrored to text file
    log_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\training_logs"
    sys.stdout = SequentialLogger(log_dir)

    save_dir = config['train']['save_dir']
    vis_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\visual_progress"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Initialize VAE3D model for RTX 5090
    model = VAE3D(in_channels=config['model']['in_channels'],
                  out_channels=config['model']['out_channels'],
                  latent_dim=config['model']['latent_dim']).to(device)

    ckpt_path, start_epoch = find_latest_checkpoint(save_dir)
    if ckpt_path:
        print(f"RESUMING FROM: {ckpt_path} (EPOCH {start_epoch})")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("STARTING NEW TRAINING SESSION")

    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    criterion = VAELoss(config=config, kld_weight=0.005).to(device)
    bce_stable = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    scaler = GradScaler('cuda')

    # Data Loading using tidied config keys
    dataset = NiftiDataset(img_dir=config['data']['raw_data_root'],
                           list_path=config['data']['training_ids'],
                           image_size=config['model']['image_size'], is_train=True)
    loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=0)

    rolling_dice = 0.0

    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        epoch_dice, epoch_loss, count = 0, 0, 0

        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            img, mask = batch['image'].to(device), (batch['mask'].to(device) > 0.5).float()

            # Extract Subject ID from the batch dictionary
            subject_id = batch.get('id', ['unknown'])[0]

            corr = morphological_corruption(mask)

            with autocast('cuda'):
                # 2-Channel input: MRI + Corrupted Mask
                recon, mu, logvar = model(torch.cat([img, corr], dim=1))
                d_acc = (2. * (torch.sigmoid(recon) * mask).sum()) / (torch.sigmoid(recon).sum() + mask.sum() + 1e-6)

                # Phase logic using thresholds: 0.80, 0.875, 0.90, 0.95
                if rolling_dice < 0.80:
                    loss = bce_stable(recon, mask) + (1.0 - d_acc)
                else:
                    loss, _, _, _ = criterion(recon, mask, mu, logvar, corrupted_input=corr, calculate_boundary=True)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_dice += d_acc.item();
            epoch_loss += loss.item();
            count += 1

            # Save visual check with ID every 10 epochs
            if i == 0 and (epoch + 1) % 10 == 0:
                save_visual_check(recon, mask, img, epoch + 1, vis_dir, subject_id)

        rolling_dice = epoch_dice / count
        # Dynamic phase update from tidied YAML config
        current_phase = update_weights_from_config(criterion, rolling_dice, config)

        print(f"Epoch {epoch + 1:03d} | {current_phase} | Dice: {rolling_dice:.4f} | Loss: {epoch_loss / count:.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch + 1}.pth"))


if __name__ == "__main__":
    train_vae()