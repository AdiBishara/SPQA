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
from utils.models.vae import VAE3D
from losses.losses import DAELoss, dice_coefficient
import torch.nn.functional as F


# --- Data Augmentation & Corruption ---

def morphological_corrupt(mask, severity=0.0):
    """Dynamically scales morphological degradation and slab masking based on training phase severity."""
    corr = mask.clone()

    # --- Stage 1: Morphological corruption ---
    # As severity increases, we apply more iterations (from 1 up to 5)
    max_iters = 1 + int(severity * 4)  
    iterations = torch.randint(1, max_iters + 1, (1,)).item()
    
    # We randomly choose to either erode or dilate the mask, modifying the boundary
    if torch.rand(1).item() > 0.5:
        # Erosion (shrinks the shape)
        for _ in range(iterations):
            corr = F.max_pool3d(corr, kernel_size=3, stride=1, padding=1)
    else:
        # Dilation (expands the shape)
        for _ in range(iterations):
            corr = 1.0 - F.max_pool3d(1.0 - corr, kernel_size=3, stride=1, padding=1)

    # --- Stage 2: Slab masking ---
    # We only start aggressively removing entire chunks (slabs) after the grace period.
    if severity > 0.0:
        # The likelihood of destroying a slab increases as severity goes up
        slab_prob = severity * 0.65
        if torch.rand(1).item() < slab_prob:
            # Pick a random spatial axis (2=Depth, 3=Height, 4=Width)
            axis = torch.randint(2, 5, (1,)).item()
            
            # Decide how much of the volume to remove (up to 80% at max severity)
            max_removal = 0.20 + severity * 0.60
            removal_frac = torch.FloatTensor(1).uniform_(0.15, max_removal).item()
            
            size = corr.shape[axis]
            n_remove = max(1, int(size * removal_frac))
            
            # Find a random starting position for our removal "cut"
            start = torch.randint(0, max(1, size - n_remove + 1), (1,)).item()
            
            # Build the slicing dynamically and apply it
            slices = [slice(None)] * 5
            slices[axis] = slice(start, start + n_remove)
            corr[tuple(slices)] = 0.0

    return corr


def augment_3d(image, mask):
    """Applies consistent 3D spatial augmentations (flips, affine transformations) to image and mask."""
    # --- 1. Random Flipping ---
    # We iterate over Depth, Height, and Width
    for dim in [2, 3, 4]:
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, [dim])
            mask = torch.flip(mask, [dim])

    # --- 2. Random Affine (Rotation + Scale + Offset) ---
    # Pick random angles in radians
    ax = torch.FloatTensor(1).uniform_(-30, 30).item() * (np.pi / 180)
    ay = torch.FloatTensor(1).uniform_(-30, 30).item() * (np.pi / 180)
    az = torch.FloatTensor(1).uniform_(-30, 30).item() * (np.pi / 180)

    # Build the rotation matrices for X, Y, and Z
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)

    # Combine rotations and apply a random scale factor
    scale = torch.FloatTensor(1).uniform_(0.7, 1.4).item()
    R = (Rz @ Ry @ Rx) * scale

    # Shift the image slightly (up to 10% translation)
    shift = torch.FloatTensor(3).uniform_(-0.1, 0.1)

    # Construct the final affine matrix
    B = image.shape[0]
    theta = torch.zeros(B, 3, 4)
    theta[:, :3, :3] = R
    theta[:, :, 3] = shift

    # Apply the transformation grid
    grid = F.affine_grid(theta.to(image.device), image.shape, align_corners=False)
    
    # We use bilinear for the smooth MRI image, but nearest-neighbor for the binary mask
    image = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    mask = F.grid_sample(mask, grid, mode='nearest', padding_mode='zeros', align_corners=False)

    return image, mask


# --- Training Stability Constants ---
CONFIRM_EPOCHS = 5    # Consecutive epochs required above threshold to advance phase.
BLEND_EPOCHS = 30     # Epoch duration for smoothing loss weight transitions.
EMA_ALPHA = 0.02      # Smoothing factor for Dice score EMA.

class PhaseManager:
    """Manages curriculum phases using performance hysteresis and smooth weight blending."""

    def __init__(self, config):
        # We extract our phase definitions from the config YAML
        phases_cfg = config['phases']
        self.phases = []
        for name, data in phases_cfg.items():
            weights = {k: v for k, v in data.items() if k != 'threshold'}
            self.phases.append((name, data['threshold'], weights))

        # We start at the easiest phase (index 0)
        self.locked_idx = 0
        self.confirm_counter = 0

        # Blending State (used when transitioning between phases)
        self.blending = False
        self.blend_epoch = 0
        self.prev_weights = None
        self.target_weights = None

    @property
    def current_name(self):
        """Returns the human-readable string name of our current phase."""
        return self.phases[self.locked_idx][0]

    def get_weights(self):
        """Returns the active loss weights, seamlessly handling mid-phase blends."""
        if not self.blending:
            return dict(self.phases[self.locked_idx][2])

        # If we are blending, we calculate the alpha step
        alpha = min(self.blend_epoch / BLEND_EPOCHS, 1.0)
        blended = {}
        for key in self.target_weights:
            old_val = self.prev_weights.get(key, 0.0)
            new_val = self.target_weights[key]
            # Standard linear interpolation
            blended[key] = old_val + alpha * (new_val - old_val)
        return blended

    def step(self, ema_dice):
        """
        Evaluates the current moving average of our performance. If it's high enough
        consistently, we trigger an advancement to the next phase.
        """
        status = None

        # Process blending if active
        if self.blending:
            self.blend_epoch += 1
            alpha = min(self.blend_epoch / BLEND_EPOCHS, 1.0)
            status = (f"    >>> BLENDING: {self.phases[self.locked_idx - 1][0]} -> "
                      f"{self.current_name} [epoch {self.blend_epoch}/{BLEND_EPOCHS}, "
                      f"a={alpha:.2f}]")
            
            # Stop blending once we reach the top
            if self.blend_epoch >= BLEND_EPOCHS:
                self.blending = False
                status = (f"    >>> BLEND COMPLETE: fully running in {self.current_name}")

        # Review performance for potential advancement
        next_idx = self.locked_idx + 1
        if next_idx < len(self.phases):
            next_threshold = self.phases[next_idx][1]
            
            # Are we doing well enough to move on?
            if ema_dice >= next_threshold:
                self.confirm_counter += 1
                if self.confirm_counter >= CONFIRM_EPOCHS:
                    # Excellent! Lock in the new phase and begin blending.
                    self.prev_weights = dict(self.phases[self.locked_idx][2])
                    self.locked_idx = next_idx
                    self.target_weights = dict(self.phases[self.locked_idx][2])
                    self.confirm_counter = 0
                    self.blending = True
                    self.blend_epoch = 0
                    status = (f"    >>> PHASE LOCKED: {self.current_name} "
                              f"(Confirmed stable performance above {next_threshold:.2f})")
                else:
                    pending = (f"    >>> Phase advance pending... "
                               f"({self.confirm_counter}/{CONFIRM_EPOCHS} epochs)")
                    status = status + "\n" + pending if status else pending
            else:
                # If performance drops below the threshold, reset the counter
                if self.confirm_counter > 0:
                    status_reset = (f"    >>> Advance counter reset "
                                    f"(EMA dice dropped below {next_threshold:.2f})")
                    status = status + "\n" + status_reset if status else status_reset
                self.confirm_counter = 0

        return self.current_name, status


# --- Utilities ---

def find_latest_checkpoint(save_dir):
    """Locates the highest epoch checkpoint in the given directory."""
    checkpoints = glob.glob(os.path.join(save_dir, "vae_epoch_*.pth"))
    if not checkpoints: 
        return None, 0
        
    def extract_epoch(ckpt_path):
        m = re.search(r'vae_epoch_(\d+).pth', ckpt_path)
        return int(m.group(1)) if m else 0
        
    latest = max(checkpoints, key=extract_epoch)
    return latest, extract_epoch(latest)


class SequentialLogger(object):
    """T-ees stdout to a sequentially numbered text file to preserve run history."""
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.terminal = sys.stdout

        # Figure out what run number we are currently on
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
        
        banner = f"--- SPQA VAE TRAINING SESSION {num} ---\n"
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
    """Exports a 3x4 slice grid comparing corrupted input, ground truth, and VAE prediction."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        d = image.shape[2]
        
        # We capture 3 specific slices along the Z-axis
        slices = [int(d*0.25), int(d*0.5), int(d*0.75)]
        slice_names = ['Inferior', 'Central', 'Superior']
        col_titles = ['Corrupted Input', 'Ground Truth', 'VAE Prediction', 'Repair Task']

        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle(f"VAE Visual Validation | Epoch {epoch:04d} | Subject: {sid}", fontsize=16, fontweight='bold')

        for i, (s_idx, s_name) in enumerate(zip(slices, slice_names)):
            img_s = image[0,0,s_idx].detach().cpu().numpy()
            corr_np = corr[0,0,s_idx].detach().cpu().numpy()
            tgt_np = target[0,0,s_idx].detach().cpu().numpy()
            pred_prob = torch.sigmoid(recon[0,0,s_idx]).detach().cpu().numpy()
            pred_mask = (pred_prob > 0.40).astype(np.float32)

            # 1. Overlay the generated corruption (yellow)
            axes[i,0].imshow(img_s, cmap='gray')
            if np.any(corr_np):
                axes[i,0].imshow(np.ma.masked_where(corr_np < 0.5, corr_np), cmap='autumn', alpha=0.50, interpolation='nearest')

            # 2. Overlay the pristine ground truth (green)
            axes[i,1].imshow(img_s, cmap='gray')
            if np.any(tgt_np):
                axes[i,1].imshow(np.ma.masked_where(tgt_np < 0.5, tgt_np), cmap='Greens', alpha=0.50, interpolation='nearest')

            # 3. Overlay what the VAE *thinks* it looks like (red)
            axes[i,2].imshow(img_s, cmap='gray')
            if np.any(pred_mask):
                axes[i,2].imshow(np.ma.masked_where(pred_mask < 0.5, pred_mask), cmap='Reds', alpha=0.50, interpolation='nearest')

            # 4. Difference Heatmap: displays the literal diff the model needs to fix.
            diff = tgt_np - corr_np
            axes[i,3].imshow(img_s, cmap='gray')
            axes[i,3].imshow(diff, cmap='RdBu', alpha=0.65, vmin=-1, vmax=1, interpolation='nearest')

            # Add human readable titles for context
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
        print(f"Visualizing failed, continuing anyway. Error: {e}")


# --- Main Training Routine ---

def train_vae():
    """Initializes the VAE environment and executes the training loop."""
    
    # Configuration and Environment Setup
    config = load_config(r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\vae_config.yaml")
    fix_seeds(config['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log_dir = os.path.join(r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs", "training_logs")
    save_dir = os.path.join(r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs", "vae_checkpoints")
    vis_dir = os.path.join(r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs", "visual_progress")
    
    for d in [log_dir, save_dir, vis_dir]: 
        os.makedirs(d, exist_ok=True)
        
    sys.stdout = SequentialLogger(log_dir)
    
    # Initialize the 3D-VAE Architecture
    model = VAE3D(
        in_channels=config['model'].get('in_channels', 2),
        out_channels=config['model'].get('out_channels', 1),
        latent_dim=config['model']['latent_dim']
    ).to(device)
    
    # Reload weights from the last checkpoint if one exists
    ckpt, start_epoch = find_latest_checkpoint(save_dir)
    if ckpt: 
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
            print(f"Successfully resumed from checkpoint: {ckpt}")
        except Exception as e:
            print(f"Warning: Corrupt checkpoint {ckpt}. Starting fresh. Error: {e}")
            start_epoch = 0

    # Set up optimizer, scaler, and dataloader
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=1e-5)
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

    # Phase management logic to direct curriculum learning
    phase_mgr = PhaseManager(config)
    ema_dice = 0.0          
    max_ema_achieved = 0.0  
    ema_initialized = False  

    print("Beginning Training Loop...")
    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        
        # Reset trackers for this epoch
        metrics = {'dice': 0.0, 'bce': 0.0, 'contour': 0.0, 'kld': 0.0, 'count': 0}

        # Synchronize our loss criteria with the phase manager
        criterion.w = phase_mgr.get_weights()

        for i, batch in enumerate(loader):
            img = batch['image'].to(device)
            # Threshold ground truth strictly just in case
            gt = (batch['gt'].to(device) > 0.5).float()
            
            # Apply our spatial augmentations
            img, gt = augment_3d(img, gt)
            
            # Identify current corruption difficulty and shatter the input
            severity = criterion.w.get('corruption_severity', 0.0)
            corr = morphological_corrupt(gt, severity)
            
            # Forward pass wrapped in mixed precision for speed!
            with autocast('cuda'):
                recon, mu, logvar = model(torch.cat([img, corr], dim=1))
                
                # Check dice similarity
                d_acc = dice_coefficient(torch.sigmoid(recon), gt)
                
                # Compute total combined loss
                loss, _, contour_val, bce_val, kld_val = criterion(recon, gt, mu, logvar)
            
            # If the loss blows up, we discard the batch instead of dying
            if torch.isnan(loss) or torch.isnan(d_acc):
                optimizer.zero_grad()
                continue
                
            # Standard backward operations
            scaler.scale(loss).backward()
            
            # Unscale before clipping to ensure math checks out correctly
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
                
            # Log metrics for our end-of-epoch report
            metrics['dice'] += d_acc.item() 
            metrics['bce'] += bce_val.item() 
            metrics['contour'] += contour_val.item() 
            metrics['kld'] += kld_val.item()
            metrics['count'] += 1
            
            # Save visual debug grids every 10 epochs using the first batch
            if i == 0 and (epoch + 1) % 10 == 0: 
                sid = batch['id'][0]
                save_visual_check(recon, corr, gt, img, epoch+1, vis_dir, sid)

        # Ensure we didn't just crash out of an entire epoch
        if metrics['count'] == 0:
            print(f"Epoch {epoch+1:03d} | SKIPPED (All batches produced NaN)")
            continue

        # Smooth out our Dice performance reporting
        epoch_dice = metrics['dice'] / metrics['count']
        if not ema_initialized:
            ema_dice = epoch_dice
            ema_initialized = True
        else:
            ema_dice = EMA_ALPHA * epoch_dice + (1 - EMA_ALPHA) * ema_dice
            
        if epoch > 10:
            max_ema_achieved = max(max_ema_achieved, ema_dice)

        # Feed performance back to phase manager
        current_phase, phase_status = phase_mgr.step(ema_dice)

        # Output detailed epoch findings
        n = metrics['count']
        print(f"Epoch {epoch+1:03d} | {current_phase} | Dice: {epoch_dice:.4f} | EMA: {ema_dice:.4f} | BCE: {metrics['bce']/n:.4f}")
        print(f"    > Breakdown: Contour: {metrics['contour']/n:.4f} | KLD: {metrics['kld']/n:.4f}")
        if phase_status:
            print(phase_status)
            
        # Snapshot the model weights
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train_vae()
