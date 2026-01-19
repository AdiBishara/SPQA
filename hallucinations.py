import os
import torch
import numpy as np
import nibabel as nib
from monai.metrics import DiceMetric
from utils.config import load_config
from utils.models.vae import VAE3D
import glob
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
ARTIFACT_SIZE = 30  # Size of the fake "error" box (voxels)
ARTIFACT_LOC = (128, 128, 128)  # Center of the brain


def get_latest_checkpoint(save_dir, prefix="vae_epoch_"):
    checkpoints = glob.glob(os.path.join(save_dir, f"{prefix}*.pth"))
    if not checkpoints: return None
    latest_ckpt = max(checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    return latest_ckpt


def add_artifact(mask_tensor):
    """Draws a solid white box into the mask to simulate a terrible segmentation error."""
    bad_mask = mask_tensor.clone()
    # Add a 3D block of 1s (False Positive Artifact)
    x, y, z = ARTIFACT_LOC
    s = ARTIFACT_SIZE // 2
    bad_mask[:, :, x - s:x + s, y - s:y + s, z - s:z + s] = 1.0
    return bad_mask


def run_anomaly_test():
    print("--- STARTING ANOMALY DETECTION TEST ---")
    config_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE (In=2, Out=1)
    print("Loading VAE...")
    model = VAE3D(
        in_channels=2,
        out_channels=1,
        image_size=config['model']['image_size'],
        latent_dim=2048
    ).to(device)

    ckpt_path = get_latest_checkpoint(config['QC']['checkpoint_dir'])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Load ONE Test Subject
    with open(config['Data']['test_ids'], 'r') as f:
        subject_id = f.readline().strip()  # Just take the first one

    print(f"Testing on Subject: {subject_id}")
    base_path = os.path.join(config['Data']['raw_data_root'], subject_id)

    # Load Data
    n_img = nib.load(os.path.join(base_path, "image.nii.gz") if os.path.exists(
        os.path.join(base_path, "image.nii.gz")) else os.path.join(base_path, "image.nii"))
    n_mask = nib.load(os.path.join(base_path, "mask.nii.gz") if os.path.exists(
        os.path.join(base_path, "mask.nii.gz")) else os.path.join(base_path, "mask.nii"))

    img = n_img.get_fdata().astype(np.float32)
    mask = n_mask.get_fdata().astype(np.float32)

    # Normalize & Pad
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    target_size = (256, 256, 256)
    c_img = np.zeros(target_size, dtype=np.float32)
    c_mask = np.zeros(target_size, dtype=np.float32)

    d, h, w = mask.shape
    z_off, y_off, x_off = (256 - d) // 2, (256 - h) // 2, (256 - w) // 2

    c_img[z_off:z_off + d, y_off:y_off + h, x_off:x_off + w] = img
    c_mask[z_off:z_off + d, y_off:y_off + h, x_off:x_off + w] = mask

    t_img = torch.from_numpy(c_img).unsqueeze(0).unsqueeze(0).to(device)
    t_mask = torch.from_numpy(c_mask).unsqueeze(0).unsqueeze(0).to(device)

    # --- EXPERIMENT 1: GOOD MASK ---
    input_good = torch.cat([t_img, t_mask], dim=1)

    with torch.no_grad():
        recon_good_logits, _, _ = model(input_good)
        recon_good = (torch.sigmoid(recon_good_logits) > 0.5).float()

        dice_metric = DiceMetric(include_background=False, reduction="mean")
        dice_metric(y_pred=recon_good, y=t_mask)
        score_good = dice_metric.aggregate().item()
        dice_metric.reset()

    # --- EXPERIMENT 2: BAD MASK (Simulated Error) ---
    t_mask_bad = add_artifact(t_mask)
    input_bad = torch.cat([t_img, t_mask_bad], dim=1)

    with torch.no_grad():
        recon_bad_logits, _, _ = model(input_bad)
        recon_bad = (torch.sigmoid(recon_bad_logits) > 0.5).float()

        # KEY: We compare the VAE output vs. the BAD Input
        # If the VAE "Fixed" the error, it will disagree with the Bad Input -> Low Dice
        dice_metric(y_pred=recon_bad, y=t_mask_bad)
        score_bad = dice_metric.aggregate().item()
        dice_metric.reset()

    print("\n" + "=" * 40)
    print(f"RESULTS FOR ANOMALY DETECTION:")
    print(f"1. Score on GOOD Mask:  {score_good:.4f} (Baseline)")
    print(f"2. Score on BAD Mask:   {score_bad:.4f}  (With Artifact)")
    print("-" * 40)

    gap = score_good - score_bad
    print(f"DELTA (Sensitivity):    {gap:.4f}")

    if gap > 0.05:
        print("✅ SUCCESS: The VAE detected the error (Score dropped).")
    else:
        print("❌ FAILURE: The VAE hallucinated the artifact (Score stayed high).")
    print("=" * 40)


if __name__ == "__main__":
    run_anomaly_test()