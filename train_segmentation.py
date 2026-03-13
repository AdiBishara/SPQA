import os
import sys
import glob
import re
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.seeding import fix_seeds
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.models.unet_dropout import UNet

# --- NVIDIA Hardware Optimizations ---
# Enable TF32 matrix math for significant speedups on Ampere GPUs.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --- Data Augmentation & Loss Definition ---

def gpu_augment(images, masks):
    """Executes high-speed random 3D flips and 90-degree rotations purely on the GPU."""
    # 1. Random Flip
    if torch.rand(1, device=images.device).item() > 0.5:
        # Pick a random spatial dimension to flip
        axis = torch.randint(2, 5, (1,), device=images.device).item()
        images = torch.flip(images, [axis])
        masks = torch.flip(masks, [axis])

    # 2. Random 90-degree Rotation
    if torch.rand(1, device=images.device).item() > 0.5:
        # Choose how many 90-degree turns (1, 2, or 3)
        k = torch.randint(1, 4, (1,), device=images.device).item()
        
        # Pick the plane of rotation (e.g. dimensions 3 and 4 for X-Y plane)
        dims = torch.randint(2, 5, (2,), device=images.device).tolist()
        
        if dims[0] != dims[1]:
            images = torch.rot90(images, k, dims)
            masks = torch.rot90(masks, k, dims)

    return images, masks


def robust_dice_loss(pred, target, smooth=1e-5):
    """Computes continuous Dice loss with smoothing to prevent zero-division errors."""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # We return 1.0 - dice since we want to *minimize* the loss
    return 1.0 - dice


# --- Logging & Checkpoint Utilities ---

class HybridLogger(object):
    """Custom logger that caches full traces to disk but filters console output for readability."""
    def __init__(self, filepath, resume=False):
        mode = "a" if resume else "w"
        self.log = open(filepath, mode, encoding='utf-8')
        self.terminal = sys.stdout

    def write(self, message):
        # 1. Always record the exact character sequence to disk
        self.log.write(message)
        self.log.flush()

        # 2. Only echo specific, important keywords to the human operator's terminal
        clean_keywords = ["Epoch", "Segmentation Dice Loss", "Loading", "Starting", "Saved"]

        # We also pass through blank lines for formatting
        if any(k in message for k in clean_keywords) or message.strip() == "":
            self.terminal.write(message)
            self.terminal.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_latest_checkpoint(save_dir):
    """Scans the designated directory to find the highest-epoch checkpoint filename."""
    checkpoints = glob.glob(os.path.join(save_dir, "unet3d_epoch_*.pth"))
    if not checkpoints: 
        return None, 0
        
    latest_ckpt = max(checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    start_epoch = int(re.search(r'epoch_(\d+)', latest_ckpt).group(1))
    
    return latest_ckpt, start_epoch


def get_log_file(base_dir, resume_run=None):
    """Generates a sequentially incremented log filename to protect existing run records."""
    os.makedirs(base_dir, exist_ok=True)
    
    if resume_run: 
        return os.path.join(base_dir, f"segmentation_run_{resume_run}.txt")
        
    existing_logs = glob.glob(os.path.join(base_dir, "segmentation_run_*.txt"))
    max_run = 0
    
    for log_file in existing_logs:
        try:
            r = int(os.path.basename(log_file).split('_')[-1].split('.')[0])
            if r > max_run: max_run = r
        except ValueError:
            continue
            
    return os.path.join(base_dir, f"segmentation_run_{max_run + 1}.txt")


# --- Main Training Routine ---

def train_unet():
    """Initializes the UNet environment and handles the entire training process."""
    
    # Setup configuration and hardware logic
    config_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml"
    config = load_config(config_path)
    fix_seeds(config['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Establish directories
    save_dir = config['Train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    log_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\training_logs"

    # Attempt to locate an existing checkpoint to resume from
    latest_ckpt, start_epoch = get_latest_checkpoint(save_dir)

    # Initialize logging
    if latest_ckpt:
        log_file = get_log_file(log_dir)
        sys.stdout = HybridLogger(log_file, resume=True)
    else:
        log_file = get_log_file(log_dir)
        sys.stdout = HybridLogger(log_file)
        start_epoch = 0

    print(f"\n--- Starting High-Throughput Training on {device} ---")
    print("✅ GPU Acceleration Active: TF32 and in-memory GPU Augmentation enabled")

    # Initialize Dataset and Dataloader
    dataset = NiftiFewShotDataset(
        data_root=config['Data']['raw_data_root'],
        id_file=config['Data']['training_ids'],
        image_size=config['model']['image_size'],
        is_train=True
    )

    loader = DataLoader(
        dataset,
        batch_size=config['Train']['batch_size'],
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )
    print("Data successfully mapped and loaded.")

    # Initialize the UNet Model
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['n_classes'],
        channels=config['model']['channels'],
        strides=config['model']['strides'],
        dropout=config['model']['dropout_rate'],
        spatial_dims=3
    ).to(device)

    # We use AMP (Automatic Mixed Precision) via GradScaler for 2x memory/speed efficiency
    optimizer = optim.Adam(model.parameters(), lr=config['Train']['learning_rate_gen'])
    scaler = torch.amp.GradScaler('cuda')

    # Load weights if we are resuming
    if latest_ckpt:
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        print(f"Model Loaded. Resuming train sequence from Epoch {start_epoch}.")

    model.train()

    print("Beginning Training Loop...")
    for epoch in range(start_epoch, config['Train']['epochs']):
        epoch_loss = 0
        valid_batches = 0

        for batch_idx, (images, masks, _) in enumerate(loader):
            # Move tensors to GPU asynchronously
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Apply our insanely fast GPU augmentations
            images, masks = gpu_augment(images, masks)

            optimizer.zero_grad()

            # Execute forward pass in Mixed Precision
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                
                # Squeeze the logits into probabilities [0, 1]
                probs = torch.sigmoid(outputs)
                
                # Clamp probabilities strictly to avoid NaN in log space/math issues
                probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
                
                # Calculate the difference
                loss = robust_dice_loss(probs, masks)

            # Safety net: Skip if gradients explode into NaN or Infinity
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # Execute backward pass via GradScaler
            scaler.scale(loss).backward()
            
            # Unscale to properly clip gradients before the optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            valid_batches += 1

        # Calculate epoch average
        div = valid_batches if valid_batches > 0 else 1
        avg_loss = epoch_loss / div

        print(f"Epoch {epoch + 1:04d} | Segmentation Dice Loss: {avg_loss:.4f}")

        # Snapshot weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"unet3d_epoch_{epoch + 1}.pth"))
            print(f"   -> Checkpoint Saved (Epoch {epoch + 1})")


if __name__ == "__main__":
    train_unet()