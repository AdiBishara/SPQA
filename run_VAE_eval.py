import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import sys
import re
from torch.utils.data import DataLoader

# Check for MONAI
try:
    from monai.metrics import DiceMetric, HausdorffDistanceMetric

    HAS_MONAI = True
except ImportError:
    print("Warning: MONAI not found. Performance metrics will be limited to Dice.")
    HAS_MONAI = False

from utils.config import load_config
from utils.models.vae import VAE3D
from utils.data.nifti_loader import NiftiDataset
from losses.losses import get_3d_active_contour_length, dice_coefficient

# Force output to console immediately
sys.stdout.reconfigure(encoding='utf-8')


# --- 0. RUN DIRECTORY MANAGER ---
def get_run_directory(base_results_path, project_name):
    """
    Creates a versioned folder: [ProjectName]_run_[X]
    """
    os.makedirs(base_results_path, exist_ok=True)
    pattern = re.compile(rf"{re.escape(project_name)}_run_(\d+)")

    existing_runs = []
    for d in os.listdir(base_results_path):
        match = pattern.match(d)
        if match:
            existing_runs.append(int(match.group(1)))

    next_run = max(existing_runs) + 1 if existing_runs else 1
    run_folder_name = f"{project_name}_run_{next_run}"
    run_path = os.path.join(base_results_path, run_folder_name)

    os.makedirs(run_path, exist_ok=True)
    return run_path, run_folder_name


# --- 1. EVALUATION ENGINE ---
def run_vae_evaluation():
    # --- CONFIGURATION ---
    # SET YOUR SPECIFIC CHECKPOINT HERE
    CHECKPOINT_TO_USE = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\vae_checkpoints\vae_epoch_230.pth"

    config_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml"
    config = load_config(config_path)
    project_name = config.get('project_name', 'SPQA_FullVol_Ultra')

    base_results_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\results_vae_test"
    run_dir, run_name = get_run_directory(base_results_path, project_name)

    # Load Subject IDs directly from the text file
    with open(config['Data']['test_ids'], 'r') as f:
        subject_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"\n--- STARTING EVALUATION RUN: {run_name} ---")
    print(f"Loading Weights: {os.path.basename(CHECKPOINT_TO_USE)}")
    print(f"Target Directory: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. LOAD MODEL ---
    model = VAE3D(in_channels=2, out_channels=1, latent_dim=2048).to(device)
    if os.path.exists(CHECKPOINT_TO_USE):
        model.load_state_dict(torch.load(CHECKPOINT_TO_USE, map_location=device))
    else:
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_TO_USE}")
        return
    model.eval()

    # --- 3. DATASET ---
    test_dataset = NiftiDataset(
        img_dir=config['Data']['raw_data_root'],
        list_path=config['Data']['test_ids'],
        image_size=config['model']['image_size'],
        is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # --- 4. PROCESSING ---
    results = []
    affine = np.eye(4)  # Identity affine for ITK-SNAP alignment

    if HAS_MONAI:
        dice_m = DiceMetric(include_background=False, reduction="mean")
        hd95_m = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    print(f"Processing {len(test_dataset)} subjects...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Get specific subject name from our list
            subject_id = subject_names[i]

            img = batch['image'].to(device)
            mask = (batch['mask'].to(device) > 0.5).float()

            # Inference
            recon, _, _ = model(torch.cat([img, mask], dim=1))
            recon_probs = torch.sigmoid(recon)
            recon_mask = (recon_probs > 0.5).float()

            # Metrics
            if HAS_MONAI:
                dice_m(y_pred=recon_mask, y=mask)
                hd95_m(y_pred=recon_mask, y=mask)
                d_val = dice_m.aggregate().item()
                h_val = hd95_m.aggregate().item()
                dice_m.reset()
                hd95_m.reset()
            else:
                d_val = dice_coefficient(recon_mask, mask).item()
                h_val = -1.0

            c_ratio = get_3d_active_contour_length(recon_probs).item() / (
                        get_3d_active_contour_length(mask).item() + 1e-8)

            print(f"[{i + 1}/{len(test_dataset)}] {subject_id} | Dice: {d_val:.4f}")

            results.append({
                "Subject_Name": subject_id,
                "Dice": d_val,
                "HD95": h_val,
                "Contour_Ratio": c_ratio
            })

            # Save NIfTI Files with subject-specific names
            nib.save(nib.Nifti1Image(img.cpu().numpy().squeeze(), affine),
                     os.path.join(run_dir, f"{subject_id}_MRI.nii.gz"))
            nib.save(nib.Nifti1Image(mask.cpu().numpy().squeeze(), affine),
                     os.path.join(run_dir, f"{subject_id}_GT.nii.gz"))
            nib.save(nib.Nifti1Image(recon_mask.cpu().numpy().squeeze(), affine),
                     os.path.join(run_dir, f"{subject_id}_Pred.nii.gz"))

    # --- 5. EXPORT CSV ---
    df = pd.DataFrame(results)
    csv_path = os.path.join(run_dir, f"report_{run_name}.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n✅ All files saved to: {run_dir}")
    print(f"Mean Dice for this run: {df['Dice'].mean():.4f}")


if __name__ == "__main__":
    run_vae_evaluation()