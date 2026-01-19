import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from monai.metrics import DiceMetric
from utils.config import load_config
from utils.models.vae import VAE3D
import glob
import re
import sys

# Force output to console immediately (fixes silent printing)
sys.stdout.reconfigure(encoding='utf-8')


def get_latest_checkpoint(save_dir, prefix="vae_epoch_"):
    checkpoints = glob.glob(os.path.join(save_dir, f"{prefix}*.pth"))
    if not checkpoints: return None
    latest_ckpt = max(checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    return latest_ckpt


def run_vae_evaluation():
    # 1. SETUP
    print("--- STARTING VAE EVALUATION ---")
    config_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\results_vae_test"
    os.makedirs(output_dir, exist_ok=True)

    # 2. LOAD MODEL (Confirmed Architecture: In=2, Out=1)
    print(f"Loading Model on {device}...")
    model = VAE3D(
        in_channels=2,  # Image + Mask
        out_channels=1,  # Mask Only
        image_size=config['model']['image_size'],
        latent_dim=2048
    ).to(device)

    # 3. LOAD WEIGHTS
    ckpt_dir = config['QC']['checkpoint_dir']
    ckpt_path = get_latest_checkpoint(ckpt_dir)
    print(f"Loading Weights: {os.path.basename(ckpt_path)}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 4. PREPARE DATA
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    with open(config['Data']['test_ids'], 'r') as f:
        test_ids = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Found {len(test_ids)} subjects.")

    # 5. RUN INFERENCE
    results = []

    for subject_id in test_ids:
        print(f"\nProcessing: {subject_id}")

        # Load Files
        base_path = os.path.join(config['Data']['raw_data_root'], subject_id)
        img_path = os.path.join(base_path, "image.nii.gz")
        mask_path = os.path.join(base_path, "mask.nii.gz")

        if not os.path.exists(img_path): img_path = img_path.replace(".nii.gz", ".nii")
        if not os.path.exists(mask_path): mask_path = mask_path.replace(".nii.gz", ".nii")

        # Read Nifti
        nifti_img = nib.load(img_path)
        nifti_mask = nib.load(mask_path)
        img_data = nifti_img.get_fdata().astype(np.float32)
        mask_data = nifti_mask.get_fdata().astype(np.float32)

        # Normalize Image
        if np.max(img_data) > 0:
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        # Pad to 256x256x256
        target_size = (256, 256, 256)
        canvas_img = np.zeros(target_size, dtype=np.float32)
        canvas_mask = np.zeros(target_size, dtype=np.float32)

        d, h, w = mask_data.shape
        z_off = (target_size[0] - d) // 2
        y_off = (target_size[1] - h) // 2
        x_off = (target_size[2] - w) // 2

        canvas_img[z_off:z_off + d, y_off:y_off + h, x_off:x_off + w] = img_data
        canvas_mask[z_off:z_off + d, y_off:y_off + h, x_off:x_off + w] = mask_data

        # Tensor Setup (Batch=1, Channel=1)
        t_img = torch.from_numpy(canvas_img).unsqueeze(0).unsqueeze(0).to(device)
        t_mask = torch.from_numpy(canvas_mask).unsqueeze(0).unsqueeze(0).to(device)

        # Concatenate for Input (Batch=1, Channel=2)
        input_tensor = torch.cat([t_img, t_mask], dim=1)

        # PREDICT
        with torch.no_grad():
            recon, mu, logvar = model(input_tensor)

            # Recon is already 1 channel (Mask), so we just threshold
            recon_prob = torch.sigmoid(recon)
            recon_mask = (recon_prob > 0.5).float()

            # Calculate Dice
            dice_metric(y_pred=recon_mask, y=t_mask)
            score = dice_metric.aggregate().item()
            dice_metric.reset()

            print(f"   >>> VAE RECONSTRUCTION DICE: {score:.4f}")
            results.append({"Subject": subject_id, "Dice": score})

            # Save NIfTI
            recon_numpy = recon_mask.cpu().numpy().squeeze().astype(np.uint8)
            final_recon = recon_numpy[z_off:z_off + d, y_off:y_off + h, x_off:x_off + w]
            save_name = f"{subject_id}_vae_recon.nii.gz"
            nib.save(nib.Nifti1Image(final_recon, nifti_mask.affine), os.path.join(output_dir, save_name))

    # 6. FINAL STATS
    df = pd.DataFrame(results)
    avg_dice = df['Dice'].mean()
    print("\n" + "=" * 30)
    print(f"FINAL MEAN DICE: {avg_dice:.4f}")
    print("=" * 30)

    csv_path = os.path.join(output_dir, "vae_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    run_vae_evaluation()