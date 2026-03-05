import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
import re
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.seeding import fix_seeds
from utils.data.nifti_loader import NiftiDataset
from utils.models.unet_dae import UNetDAE
from losses.losses import DAELoss, dice_coefficient
import torch.nn.functional as F


def morphological_corrupt(mask):
    """
    Randomized Morphological Corruption (Erosion or Dilation).
    - Erosion: negated 3D Max Pooling (shrinks the mask)
    - Dilation: standard 3D Max Pooling (expands the mask)
    - Severity: 1-5 random iterations
    
    This ensures the model sees a *different* corrupted shape every epoch,
    forcing it to learn how to recover the true high-frequency boundary details 
    regardless of the specific noise pattern it receives.
    """
    corr = mask.clone()
    iterations = torch.randint(1, 6, (1,)).item()

    if torch.rand(1).item() > 0.5:
        # Dilation
        for _ in range(iterations):
            corr = F.max_pool3d(corr, kernel_size=3, stride=1, padding=1)
    else:
        # Erosion (negate -> max pool -> negate)
        for _ in range(iterations):
            corr = 1.0 - F.max_pool3d(1.0 - corr, kernel_size=3, stride=1, padding=1)

    return corr

# --- STABILITY CONSTANTS ---
CONFIRM_EPOCHS = 5    # Consecutive epochs above threshold before advancing
BLEND_EPOCHS = 30     # Epochs to linearly interpolate weights during transition
EMA_ALPHA = 0.1       # Smoothing factor for rolling dice (lower = smoother)


class PhaseManager:
    """
    Manages phase transitions with:
      - Ratchet: once a phase is entered, never retreat
      - Hysteresis: require CONFIRM_EPOCHS consecutive epochs above threshold
      - Smooth blending: linearly interpolate weights over BLEND_EPOCHS
    """

    def __init__(self, config):
        # Build an ordered list of (name, threshold, weights) from config
        phases_cfg = config['phases']
        self.phases = []
        for name, data in phases_cfg.items():
            weights = {k: v for k, v in data.items() if k != 'threshold'}
            self.phases.append((name, data['threshold'], weights))

        # Current locked phase index (starts at phase 0)
        self.locked_idx = 0

        # Hysteresis counter: how many consecutive epochs above next threshold
        self.confirm_counter = 0

        # Blending state
        self.blending = False
        self.blend_epoch = 0          # Current epoch within the blend window
        self.prev_weights = None      # Weights we're blending FROM
        self.target_weights = None    # Weights we're blending TO

    @property
    def current_name(self):
        return self.phases[self.locked_idx][0]

    def get_weights(self):
        """Return the current effective weights (possibly mid-blend)."""
        if not self.blending:
            return dict(self.phases[self.locked_idx][2])

        # Linear interpolation: alpha goes 0 -> 1 over BLEND_EPOCHS
        alpha = min(self.blend_epoch / BLEND_EPOCHS, 1.0)
        blended = {}
        for key in self.target_weights:
            old_val = self.prev_weights.get(key, 0.0)
            new_val = self.target_weights[key]
            blended[key] = old_val + alpha * (new_val - old_val)
        return blended

    def step(self, ema_dice):
        """
        Called once per epoch with the EMA dice value.
        Returns (phase_name, status_message) where status_message is None
        if nothing notable happened.
        """
        status = None

        # Advance blend counter if blending
        if self.blending:
            self.blend_epoch += 1
            alpha = min(self.blend_epoch / BLEND_EPOCHS, 1.0)
            status = (f"    >>> BLENDING: {self.phases[self.locked_idx - 1][0]} -> "
                      f"{self.current_name} [epoch {self.blend_epoch}/{BLEND_EPOCHS}, "
                      f"\u03b1={alpha:.2f}]")
            if self.blend_epoch >= BLEND_EPOCHS:
                self.blending = False
                status = (f"    >>> BLEND COMPLETE: fully in {self.current_name}")

        # Check if we can advance to the next phase
        next_idx = self.locked_idx + 1
        if next_idx < len(self.phases):
            next_threshold = self.phases[next_idx][1]
            if ema_dice >= next_threshold:
                self.confirm_counter += 1
                if self.confirm_counter >= CONFIRM_EPOCHS:
                    # --- ADVANCE PHASE ---
                    self.prev_weights = dict(self.phases[self.locked_idx][2])
                    self.locked_idx = next_idx
                    self.target_weights = dict(self.phases[self.locked_idx][2])
                    self.confirm_counter = 0
                    self.blending = True
                    self.blend_epoch = 0
                    status = (f"    >>> PHASE LOCKED: {self.current_name} "
                              f"(confirmed {CONFIRM_EPOCHS} epochs above "
                              f"{next_threshold:.2f})")
                else:
                    pending = (f"    >>> Phase advance pending: "
                               f"{self.confirm_counter}/{CONFIRM_EPOCHS} "
                               f"epochs above {next_threshold:.2f}")
                    status = status + "\n" + pending if status else pending
            else:
                # Reset counter if dice drops below threshold
                if self.confirm_counter > 0:
                    status_reset = (f"    >>> Advance counter reset "
                                    f"(EMA dice {ema_dice:.4f} < {next_threshold:.2f})")
                    status = status + "\n" + status_reset if status else status_reset
                self.confirm_counter = 0

        return self.current_name, status


def find_latest_checkpoint(save_dir):
    checkpoints = glob.glob(os.path.join(save_dir, "vae_epoch_*.pth"))
    if not checkpoints: return None, 0
    def extract_epoch(ckpt_path):
        m = re.search(r'vae_epoch_(\d+).pth', ckpt_path)
        return int(m.group(1)) if m else 0
    latest = max(checkpoints, key=extract_epoch)
    return latest, extract_epoch(latest)

class SequentialLogger(object):
    """
    Logs all training stdout to a numbered file in log_dir.
    Detects the current max run number from existing files and continues from it.
    """
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.terminal = sys.stdout

        # Find the highest existing run number and continue from it
        existing = glob.glob(os.path.join(log_dir, "vae_run_*.txt"))
        if existing:
            nums = []
            for f in existing:
                m = re.search(r'vae_run_(\d+)\.txt', f)
                if m: nums.append(int(m.group(1)))
            num = max(nums) + 1 if nums else 1
        else:
            num = 1

        self.log_path = os.path.join(log_dir, f"vae_run_{num}.txt")
        self.log = open(self.log_path, "a", encoding="utf-8")
        # Write banner to both console and file
        banner = f"--- SPQA DAE TRAINING SESSION {num} ---\n"
        self.terminal.write(banner)
        self.terminal.flush()
        self.log.write(banner)
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def save_visual_check(recon, corr, target, image, epoch, save_dir, sid):
    """
    Saves a 3x4 grid visualization:
      Col 0: MRI + Corrupted Mask (yellow) - what the model receives
      Col 1: MRI + Ground Truth (green)    - the target
      Col 2: MRI + DAE Prediction (red)    - the model's output
      Col 3: Difference Heatmap (hot)      - GT minus Corrupted (what the model must repair)
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        d = image.shape[2]
        slices = [int(d*0.25), int(d*0.5), int(d*0.75)]
        slice_names = ['Inferior', 'Central', 'Superior']
        col_titles = ['Corrupted Input', 'Ground Truth', 'DAE Prediction', 'GT − Corrupted\n(repair task)']

        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle(f"DAE Visual Validation | Epoch {epoch:04d} | Subject: {sid}", fontsize=16, fontweight='bold')

        for i, (s_idx, s_name) in enumerate(zip(slices, slice_names)):
            img_s = image[0,0,s_idx].detach().cpu().numpy()
            corr_np = corr[0,0,s_idx].detach().cpu().numpy()
            tgt_np = target[0,0,s_idx].detach().cpu().numpy()
            pred_prob = torch.sigmoid(recon[0,0,s_idx]).detach().cpu().numpy()
            pred_mask = (pred_prob > 0.40).astype(np.float32)

            # Col 0: Corrupted Input
            axes[i,0].imshow(img_s, cmap='gray')
            if np.any(corr_np):
                axes[i,0].imshow(np.ma.masked_where(corr_np < 0.5, corr_np), cmap='autumn', alpha=0.50, vmin=0, vmax=1, interpolation='nearest')

            # Col 1: Ground Truth
            axes[i,1].imshow(img_s, cmap='gray')
            if np.any(tgt_np):
                axes[i,1].imshow(np.ma.masked_where(tgt_np < 0.5, tgt_np), cmap='Greens', alpha=0.50, vmin=0, vmax=1, interpolation='nearest')

            # Col 2: DAE Prediction
            axes[i,2].imshow(img_s, cmap='gray')
            if np.any(pred_mask):
                axes[i,2].imshow(np.ma.masked_where(pred_mask < 0.5, pred_mask), cmap='Reds', alpha=0.50, vmin=0, vmax=1, interpolation='nearest')

            # Col 3: Difference Heatmap (GT - Corrupted) — shows what the model must repair
            # +1 = regions in GT but not in corrupted (erosion damage / missing voxels)
            # -1 = regions in corrupted but not in GT (dilation artifacts / extra voxels)
            diff = tgt_np - corr_np
            axes[i,3].imshow(img_s, cmap='gray')
            axes[i,3].imshow(diff, cmap='RdBu', alpha=0.65, vmin=-1, vmax=1, interpolation='nearest')

            # Titles and row labels
            axes[i,0].set_title(f"{s_name} ({s_idx}) | {col_titles[0]}", fontsize=9)
            axes[i,1].set_title(col_titles[1], fontsize=9)
            axes[i,2].set_title(col_titles[2], fontsize=9)
            axes[i,3].set_title(col_titles[3], fontsize=9)

            for ax in axes[i]: ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch:04d}_{sid}.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Visual Error: {e}")

def train_vae():
    config = load_config(r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml")
    fix_seeds(config['seed']); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir, save_dir, vis_dir = [os.path.join(r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs", d) for d in ["training_logs", "vae_checkpoints", "visual_progress"]]
    for d in [log_dir, save_dir, vis_dir]: os.makedirs(d, exist_ok=True)
    sys.stdout = SequentialLogger(log_dir)
    model = UNetDAE(in_channels=2, out_channels=1, spatial_dims=3).to(device)
    ckpt, start_epoch = find_latest_checkpoint(save_dir)
    if ckpt: 
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
            print(f"Loaded checkpoint: {ckpt} (strict=False)")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {ckpt}. Starting from scratch. Error: {e}")
            start_epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    criterion = DAELoss(config=config).to(device)
    scaler = GradScaler('cuda')
    loader = DataLoader(NiftiDataset(img_dir=config['data']['raw_data_root'], list_path=config['data']['training_ids'], image_size=config['model']['image_size']), batch_size=1, shuffle=True)

    # --- STABILITY: PhaseManager + EMA ---
    phase_mgr = PhaseManager(config)
    ema_dice = 0.0          # EMA-smoothed rolling dice
    ema_initialized = False  # Bootstrap flag for first epoch

    for epoch in range(start_epoch, config['train']['epochs']):
        model.train(); m = {'dice': 0, 'bce': 0, 'contour': 0, 'kld': 0, 'count': 0}

        # Apply current (possibly blended) weights to the loss function
        criterion.w = phase_mgr.get_weights()

        for i, batch in enumerate(loader):
            img = batch['image'].to(device)
            gt = (batch['gt'].to(device) > 0.5).float()
            sid = batch['id'][0]; corr = morphological_corrupt(gt)  # Randomized erosion/dilation
            with autocast('cuda'):
                recon, mu, logvar = model(torch.cat([img, corr], dim=1))
                d_acc = dice_coefficient(torch.sigmoid(recon), gt)
                # Apply current phase weights
                loss, _, contour_val, bce_val, kld_val = criterion(recon, gt, mu, logvar)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            m['dice'] += d_acc.item(); m['bce'] += bce_val.item(); m['count'] += 1
            m['contour'] += contour_val.item(); m['kld'] += kld_val.item()
            if i == 0 and (epoch + 1) % 10 == 0: save_visual_check(recon, corr, gt, img, epoch+1, vis_dir, sid)

        # --- EMA rolling dice (Fix 3) ---
        epoch_dice = m['dice'] / m['count']
        if not ema_initialized:
            ema_dice = epoch_dice
            ema_initialized = True
        else:
            ema_dice = EMA_ALPHA * epoch_dice + (1 - EMA_ALPHA) * ema_dice

        # --- Phase transition check (Fix 1 + Fix 2) ---
        current_phase, phase_status = phase_mgr.step(ema_dice)

        n = m['count']
        print(f"Epoch {epoch+1:03d} | {current_phase} | Dice: {epoch_dice:.4f} | EMA: {ema_dice:.4f} | BCE: {m['bce']/n:.4f}")
        print(f"    > Breakdown: Contour: {m['contour']/n:.4f} | KLD: {m['kld']/n:.4f}")
        if phase_status:
            print(phase_status)
        torch.save(model.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train_vae()