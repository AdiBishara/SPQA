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

# --- NVIDIA OPTIMIZATIONS ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --- GPU AUGMENTER ---
def gpu_augment(images, masks):
    """
    Performs random 3D rotations and flips on the GPU.
    """
    if torch.rand(1, device=images.device) > 0.5:
        axis = torch.randint(2, 5, (1,), device=images.device).item()
        images = torch.flip(images, [axis])
        masks = torch.flip(masks, [axis])

    if torch.rand(1, device=images.device) > 0.5:
        k = torch.randint(1, 4, (1,), device=images.device).item()
        dims = torch.randint(2, 5, (2,), device=images.device).tolist()
        if dims[0] != dims[1]:
            images = torch.rot90(images, k, dims)
            masks = torch.rot90(masks, k, dims)

    return images, masks


# --- LOSS FUNCTION ---
def robust_dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


# --- HYBRID LOGGER (FIXED) ---
class HybridLogger(object):
    def __init__(self, filepath, resume=False):
        mode = "a" if resume else "w"
        self.log = open(filepath, mode, encoding='utf-8')
        self.terminal = sys.stdout

    def write(self, message):
        # 1. Always write everything to the file
        self.log.write(message)
        self.log.flush()

        # 2. Filter Console Output
        clean_keywords = ["Epoch", "Segmentation Dice Loss", "Loading", "Starting", "Saved"]

        # FIX: Allow message if it has a keyword OR if it's just a formatting character (newline)
        if any(k in message for k in clean_keywords) or message.strip() == "":
            self.terminal.write(message)
            self.terminal.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_latest_checkpoint(save_dir):
    checkpoints = glob.glob(os.path.join(save_dir, "unet3d_epoch_*.pth"))
    if not checkpoints: return None, 0
    latest_ckpt = max(checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    return latest_ckpt, int(re.search(r'epoch_(\d+)', latest_ckpt).group(1))


def get_log_file(base_dir, resume_run=None):
    os.makedirs(base_dir, exist_ok=True)
    if resume_run: return os.path.join(base_dir, f"segmentation_run_{resume_run}.txt")
    existing_logs = glob.glob(os.path.join(base_dir, "segmentation_run_*.txt"))
    max_run = 0
    for log_file in existing_logs:
        try:
            r = int(os.path.basename(log_file).split('_')[-1].split('.')[0])
            if r > max_run: max_run = r
        except:
            continue
    return os.path.join(base_dir, f"segmentation_run_{max_run + 1}.txt")


def train_unet():
    # 1. SETUP
    config_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml"
    config = load_config(config_path)
    fix_seeds(config['seed'])

    save_dir = config['Train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latest_ckpt, start_epoch = get_latest_checkpoint(save_dir)
    log_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\training_logs"

    if latest_ckpt:
        log_file = get_log_file(log_dir)
        sys.stdout = HybridLogger(log_file, resume=True)
    else:
        log_file = get_log_file(log_dir)
        sys.stdout = HybridLogger(log_file)
        start_epoch = 0

    print(f"\n--- Starting High-Throughput Training on {device} ---")
    print("✅ GPU Acceleration: Augmentation & TF32 Enabled")

    # 2. DATA LOADER
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
    print("Data Loaded.")

    # 3. MODEL
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['n_classes'],
        channels=config['model']['channels'],
        strides=config['model']['strides'],
        dropout=config['model']['dropout_rate'],
        spatial_dims=3
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['Train']['learning_rate_gen'])
    scaler = torch.amp.GradScaler('cuda')

    if latest_ckpt:
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        print(f"Model Loaded. Resuming from Epoch {start_epoch}.")

    # 4. TRAINING LOOP
    model.train()

    for epoch in range(start_epoch, config['Train']['epochs']):
        epoch_loss = 0
        valid_batches = 0

        for images, masks, _ in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            # GPU Augmentation
            images, masks = gpu_augment(images, masks)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
                loss = robust_dice_loss(probs, masks)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            valid_batches += 1

        div = valid_batches if valid_batches > 0 else 1
        avg_loss = epoch_loss / div

        print(f"Epoch {epoch + 1} | Segmentation Dice Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"unet3d_epoch_{epoch + 1}.pth"))
            print(f"   -> Checkpoint Saved (Epoch {epoch + 1})")


if __name__ == "__main__":
    train_unet()