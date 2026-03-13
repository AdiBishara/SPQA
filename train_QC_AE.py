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


# --- Data Corruption for Denoising ---

def morphological_corrupt(mask, epoch=0, total_epochs=2500):
    """Scales morphological degradation and slab masking linearly with training epochs for DAE."""
    # Calculate how far along the curriculum we are (peaks at 60% of total epochs)
    severity = min(epoch / max(total_epochs * 0.6, 1), 1.0)

    corr = mask.clone()

    # --- Stage 1: Morphological corruption ---
    # Ramps iterations based on severity: 2 early on -> 5 late
    max_iters = 2 + int(severity * 3)  
    iterations = torch.randint(1, max_iters + 1, (1,)).item()
    
    # 50/50 chance for erosion vs dilation
    if torch.rand(1).item() > 0.5:
        # Erosion
        for _ in range(iterations):
            corr = F.max_pool3d(corr, kernel_size=3, stride=1, padding=1)
    else:
        # Dilation
        for _ in range(iterations):
            corr = 1.0 - F.max_pool3d(1.0 - corr, kernel_size=3, stride=1, padding=1)

    # --- Stage 2: Slab masking ---
    # The chance of dropping a slab increases as severity goes up
    slab_prob = severity * 0.65  
    if torch.rand(1).item() < slab_prob:
        # Pick a random spatial axis (2=Depth, 3=Height, 4=Width)
        axis = torch.randint(2, 5, (1,)).item()  
        
        # Decide how much of the volume to remove (ramps from 30% -> 80%)
        max_removal = 0.30 + severity * 0.50    
        removal_frac = torch.FloatTensor(1).uniform_(0.20, max_removal).item()
        size = corr.shape[axis]
        n_remove = max(1, int(size * removal_frac))
        
        # Determine starting slice and blank out the region
        start = torch.randint(0, max(1, size - n_remove + 1), (1,)).item()
        slices = [slice(None)] * 5
        slices[axis] = slice(start, start + n_remove)
        corr[tuple(slices)] = 0.0

    return corr


# --- Training Stability Constants ---
CONFIRM_EPOCHS = 5    # Consecutive epochs required above threshold to advance phase.
BLEND_EPOCHS = 30     # Epoch duration for smoothing loss weight transitions.
EMA_ALPHA = 0.1       # Smoothing factor for Dice score EMA.

class PhaseManager:
    """Manages curriculum phases using performance hysteresis and smooth weight blending."""

    def __init__(self, config):
        # Extract the phase configuration thresholds and weights from the YAML
        phases_cfg = config['phases']
        self.phases = []
        for name, data in phases_cfg.items():
            weights = {k: v for k, v in data.items() if k != 'threshold'}
            self.phases.append((name, data['threshold'], weights))

        # Start locked in the easiest phase
        self.locked_idx = 0
        self.confirm_counter = 0

        # State tracking for smooth weight blending between phases
        self.blending = False
        self.blend_epoch = 0          
        self.prev_weights = None      
        self.target_weights = None    

    @property
    def current_name(self):
        """Returns the human-readable string name of our current phase."""
        return self.phases[self.locked_idx][0]

    def get_weights(self):
        """Returns the current effective loss weights (handles active blending)."""
        if not self.blending:
            return dict(self.phases[self.locked_idx][2])

        # Interpolate the weights linearly over the blending window
        alpha = min(self.blend_epoch / BLEND_EPOCHS, 1.0)
        blended = {}
        for key in self.target_weights:
            old_val = self.prev_weights.get(key, 0.0)
            new_val = self.target_weights[key]
            blended[key] = old_val + alpha * (new_val - old_val)
        return blended

    def step(self, ema_dice):
        """
        Evaluates the current smoothed Dice score at the end of every epoch.
        If it surpasses the next threshold consistently, we advance the phase.
        """
        status = None

        # Advance the blend counter if we are currently blending
        if self.blending:
            self.blend_epoch += 1
            alpha = min(self.blend_epoch / BLEND_EPOCHS, 1.0)
            status = (f"    >>> BLENDING: {self.phases[self.locked_idx - 1][0]} -> "
                      f"{self.current_name} [epoch {self.blend_epoch}/{BLEND_EPOCHS}, "
                      f"a={alpha:.2f}]")
                      
            # Turn off blending once we reach the top
            if self.blend_epoch >= BLEND_EPOCHS:
                self.blending = False
                status = (f"    >>> BLEND COMPLETE: fully in {self.current_name}")

        # Check if we are ready to advance to the next phase
        next_idx = self.locked_idx + 1
        if next_idx < len(self.phases):
            next_threshold = self.phases[next_idx][1]
            
            # Are we performing above the requirement?
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
                              f"(confirmed stable performance above {next_threshold:.2f})")
                else:
                    pending = (f"    >>> Phase advance pending: "
                               f"{self.confirm_counter}/{CONFIRM_EPOCHS} "
                               f"epochs above {next_threshold:.2f}")
                    status = status + "\n" + pending if status else pending
            else:
                # Reset the counter if performance drops back below threshold
                if self.confirm_counter > 0:
                    status_reset = (f"    >>> Advance counter reset "
                                    f"(EMA dice dropped below {next_threshold:.2f})")
                    status = status + "\n" + status_reset if status else status_reset
                self.confirm_counter = 0

        return self.current_name, status


# --- Utilities ---

def find_latest_checkpoint(save_dir):
    """Locates the highest epoch checkpoint in the given directory."""
    checkpoints = glob.glob(os.path.join(save_dir, "dae_epoch_*.pth"))
    if not checkpoints: 
        return None, 0
        
    def extract_epoch(ckpt_path):
        m = re.search(r'dae_epoch_(\d+).pth', ckpt_path)
        return int(m.group(1)) if m else 0
        
    latest = max(checkpoints, key=extract_epoch)
    return latest, extract_epoch(latest)


class SequentialLogger(object):
    """T-ees stdout to a sequentially numbered text file to preserve run history."""
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.terminal = sys.stdout

        # Find the highest existing run number and continue from it
        existing = glob.glob(os.path.join(log_dir, "dae_run_*.txt"))
        if existing:
            nums = []
            for f in existing:
                m = re.search(r'dae_run_(\d+)\.txt', f)
                if m: nums.append(int(m.group(1)))
            num = max(nums) + 1 if nums else 1
        else:
            num = 1

        self.log_path = os.path.join(log_dir, f"dae_run_{num}.txt")
        self.log = open(self.log_path, "a", encoding="utf-8")
        
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
    """Exports a 3x4 slice grid comparing corrupted input, ground truth, and DAE prediction."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        d = image.shape[2]
        
        # Taking static slices across the depth of the 3D volume
        slices = [int(d*0.25), int(d*0.5), int(d*0.75)]
        slice_names = ['Inferior', 'Central', 'Superior']
        col_titles = ['Corrupted Input', 'Ground Truth', 'DAE Prediction', 'Repair Task']

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
                axes[i,0].imshow(np.ma.masked_where(corr_np < 0.5, corr_np), cmap='autumn', alpha=0.50, interpolation='nearest')

            # Col 1: Ground Truth
            axes[i,1].imshow(img_s, cmap='gray')
            if np.any(tgt_np):
                axes[i,1].imshow(np.ma.masked_where(tgt_np < 0.5, tgt_np), cmap='Greens', alpha=0.50, interpolation='nearest')

            # Col 2: DAE Prediction
            axes[i,2].imshow(img_s, cmap='gray')
            if np.any(pred_mask):
                axes[i,2].imshow(np.ma.masked_where(pred_mask < 0.5, pred_mask), cmap='Reds', alpha=0.50, interpolation='nearest')

            # Col 3: Difference Heatmap (GT - Corrupted) — shows what the model must repair
            diff = tgt_np - corr_np
            axes[i,3].imshow(img_s, cmap='gray')
            axes[i,3].imshow(diff, cmap='RdBu', alpha=0.65, vmin=-1, vmax=1, interpolation='nearest')

            # Titles and row labels
            axes[i,0].set_title(f"{s_name} ({s_idx}) | {col_titles[0]}", fontsize=9)
            axes[i,1].set_title(col_titles[1], fontsize=9)
            axes[i,2].set_title(col_titles[2], fontsize=9)
            axes[i,3].set_title(col_titles[3], fontsize=9)

            for ax in axes[i]: 
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch:04d}_{sid}.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Visual Error: {e}")


# --- Main Training Routine ---

def train_dae():
    """Initializes the DAE environment and executes the training loop."""
    
    # Setup configuration and hardware logic
    config = load_config(r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml")
    fix_seeds(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Establish logging and checkpoint directories
    log_dir = os.path.join(r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs", "training_logs")
    save_dir = os.path.join(r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs", "dae_checkpoints")
    vis_dir = os.path.join(r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs", "visual_progress")
    
    for d in [log_dir, save_dir, vis_dir]: 
        os.makedirs(d, exist_ok=True)
        
    sys.stdout = SequentialLogger(log_dir)
    
    # Init the DAE architecture
    m_cfg = config['model']
    model = UNetDAE(
        in_channels=m_cfg.get('in_channels', 2),
        out_channels=m_cfg.get('out_channels', 1),
        channels=m_cfg.get('channels', [16, 32, 64, 128, 256]),
        strides=m_cfg.get('strides', [2, 2, 2, 2]),
        num_res_units=m_cfg.get('num_res_units', 2),
        kernel_size=m_cfg.get('kernel_size', 3),
        norm=m_cfg.get('norm', 'BATCH'),
        dropout=m_cfg.get('dropout', 0.0),
        spatial_dims=3
    ).to(device)
    
    # Try to load previous run's weights
    ckpt, start_epoch = find_latest_checkpoint(save_dir)
    if ckpt: 
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
            print(f"Successfully resumed from checkpoint: {ckpt}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {ckpt}. Starting from scratch. Error: {e}")
            start_epoch = 0
            
    # Optimizer and Dataset setup
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    criterion = DAELoss(config=config).to(device)
    scaler = GradScaler('cuda')
    
    loader = DataLoader(
        NiftiDataset(
            img_dir=config['data']['raw_data_root'], 
            list_path=config['data']['training_ids'], 
            image_size=config['model']['image_size']
        ), 
        batch_size=1, 
        shuffle=True
    )

    # Boot up the Curriculum Logic
    phase_mgr = PhaseManager(config)
    ema_dice = 0.0          
    ema_initialized = False  

    print("Beginning Training Loop...")
    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        
        # Reset trackers for this epoch
        metrics = {'dice': 0.0, 'bce': 0.0, 'contour': 0.0, 'kld': 0.0, 'count': 0}

        # Apply current phase weights to the loss function
        criterion.w = phase_mgr.get_weights()

        for i, batch in enumerate(loader):
            img = batch['image'].to(device)
            # Binary threshold the ground truth mask
            gt = (batch['gt'].to(device) > 0.5).float()
            sid = batch['id'][0]
            
            # Apply our curriculum morphological corruption
            total_epochs = config['train']['epochs']
            corr = morphological_corrupt(gt, epoch=epoch, total_epochs=total_epochs)  
            
            with autocast('cuda'):
                # DAE uses a UNet architecture but has mu/logvar mock outputs to reuse DAELoss
                recon, mu, logvar = model(torch.cat([img, corr], dim=1))
                d_acc = dice_coefficient(torch.sigmoid(recon), gt)
                
                # Apply current phase weights
                loss, _, contour_val, bce_val, kld_val = criterion(recon, gt, mu, logvar)
                
            # Execute Gradient Steps safely
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Record statistics
            metrics['dice'] += d_acc.item()
            metrics['bce'] += bce_val.item()
            metrics['count'] += 1
            metrics['contour'] += contour_val.item()
            metrics['kld'] += kld_val.item()
            
            # Generate Visual Grid at standard intervals
            if i == 0 and (epoch + 1) % 10 == 0: 
                save_visual_check(recon, corr, gt, img, epoch+1, vis_dir, sid)

        # Update the trailing performance average 
        epoch_dice = metrics['dice'] / max(metrics['count'], 1)
        if not ema_initialized:
            ema_dice = epoch_dice
            ema_initialized = True
        else:
            ema_dice = EMA_ALPHA * epoch_dice + (1 - EMA_ALPHA) * ema_dice

        # Evaluate progress
        current_phase, phase_status = phase_mgr.step(ema_dice)

        # Output detailed logging snapshot
        n = max(metrics['count'], 1)
        print(f"Epoch {epoch+1:03d} | {current_phase} | Dice: {epoch_dice:.4f} | EMA: {ema_dice:.4f} | BCE: {metrics['bce']/n:.4f}")
        print(f"    > Breakdown: Contour: {metrics['contour']/n:.4f} | KLD: {metrics['kld']/n:.4f}")
        if phase_status:
            print(phase_status)
            
        # Snapshot weights to disk
        torch.save(model.state_dict(), os.path.join(save_dir, f"dae_epoch_{epoch+1}.pth"))


if __name__ == "__main__":
    train_dae()