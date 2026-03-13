import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import sys
import re
from torch.utils.data import DataLoader

# Ensure graceful fallback if MONAI isn't installed in the evaluating environment
try:
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    HAS_MONAI = True
except ImportError:
    print("Warning: MONAI not found. Advanced metrics limited to internal Dice fallback.")
    HAS_MONAI = False

from utils.config import load_config
from utils.models.vae import VAE3D
from utils.data.nifti_loader import NiftiDataset
from losses.losses import get_3d_active_contour_length, dice_coefficient

# Force output to console seamlessly across weird bash environments
sys.stdout.reconfigure(encoding='utf-8')


# --- Dynamic Routing Utilities ---

def get_run_directory(base_results_path, project_name):
    """Creates a sequentially numbered run directory to prevent data overwrites."""
    os.makedirs(base_results_path, exist_ok=True)
    
    # Scan the directory tracking only folders matching this project's name
    pattern = re.compile(rf"{re.escape(project_name)}_run_(\d+)")
    existing_runs = [int(pattern.match(d).group(1)) for d in os.listdir(base_results_path) if pattern.match(d)]
    
    # Increment the highest number safely
    next_run = max(existing_runs) + 1 if existing_runs else 1
    
    run_path = os.path.join(base_results_path, f"{project_name}_run_{next_run}")
    os.makedirs(run_path, exist_ok=True)
    
    return run_path, f"{project_name}_run_{next_run}"


# --- Main Evaluation Loop ---

def run_vae_evaluation():
    """Evaluates VAE checkpoints, calculating metrics and exporting NIfTI visualization triads."""

    # ==========================================
    # INPUT: SELECT TARGET CHECKPOINT
    # ------------------------------------------
    # Manually point this to whichever weights you want to profile.
    # ==========================================
    CHECKPOINT_TO_USE = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\vae_checkpoints\vae_epoch_460.pth"

    # Setup Environment
    config_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml"
    config = load_config(config_path)
    base_results_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\results_vae_test"

    run_dir, run_name = get_run_directory(base_results_path, config.get('project_name', 'SPQA'))

    # Retrieve Subject Directory Keys
    with open(config['Data']['test_ids'], 'r') as f:
        subject_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"\n--- EVALUATION RUN: {run_name} ---")
    print(f"Profiling Model: {os.path.basename(CHECKPOINT_TO_USE)}")

    # Load Hardware & Model Architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE3D(in_channels=2, out_channels=1, latent_dim=2048).to(device)

    # Hotload the specific weights directly
    if os.path.exists(CHECKPOINT_TO_USE):
        model.load_state_dict(torch.load(CHECKPOINT_TO_USE, map_location=device))
    else:
        print(f"❌ Error: Checkpoint string is broken or absent at {CHECKPOINT_TO_USE}")
        return
        
    # Lock model against training
    model.eval()

    # We reuse the NiftiDataset loader in evaluation mode 
    # to maintain perfectly identical preprocessing pipelines
    test_dataset = NiftiDataset(
        img_dir=config['Data']['raw_data_root'],
        list_path=config['Data']['test_ids'],
        image_size=config['model']['image_size'],
        is_train=False
    )
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = []
    
    # We strip spatial mapping here so the 3 NIfTI files snap together natively
    affine = np.eye(4)
    
    # Prep Medical Metrics
    if HAS_MONAI:
        dice_m = DiceMetric(include_background=False, reduction="mean")
        hd95_m = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    # Disable gradient tracking across the whole loop to maximize VRAM
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Resolve subject name string mapping gracefully
            sub_id = subject_names[i] if i < len(subject_names) else f"Sub_{i}"
            
            img = batch['image'].to(device)
            mask = (batch['mask'].to(device) > 0.5).float()

            # --- Inference Task ---
            # VAE takes the MRI and Mask concatenated
            recon, _, _ = model(torch.cat([img, mask], dim=1))
            
            # Compress logits into raw probabilities
            recon_probs = torch.sigmoid(recon)
            
            # Snap to binary map
            recon_mask = (recon_probs > 0.5).float()

            # --- Extract Analysis Metrics ---
            
            # 1. Dice Score
            d_val = dice_coefficient(recon_mask, mask).item()
            h_val = -1.0
            
            # Fallback to pure MONAI if it's available for higher fidelity reporting
            if HAS_MONAI:
                dice_m(y_pred=recon_mask, y=mask)
                hd95_m(y_pred=recon_mask, y=mask)
                d_val = dice_m.aggregate().item()
                h_val = hd95_m.aggregate().item()
                dice_m.reset()
                hd95_m.reset()

            # 2. Contour Length Differential
            # Calculates how fragmented the prediction is versus the smooth continuous 
            # biological structure of the genuine ground truth mask.
            c_ratio = get_3d_active_contour_length(recon_probs).item() / (
                        get_3d_active_contour_length(mask).item() + 1e-8)
                        
            print(f"[{i + 1}/{len(test_dataset)}] {sub_id} | Dice: {d_val:.4f}")

            # Persist metrics directly
            results.append({"ID": sub_id, "Dice": d_val, "HD95": h_val, "Contour": c_ratio})

            # --- Export ITK-SNAP Trio ---
            # Saving the base MRI alongside the two masks makes visual troubleshooting
            # incredibly fast using medical image viewers.
            nib.save(nib.Nifti1Image(img.cpu().numpy().squeeze(), affine),
                     os.path.join(run_dir, f"{sub_id}_MRI.nii.gz"))
            nib.save(nib.Nifti1Image(mask.cpu().numpy().squeeze(), affine),
                     os.path.join(run_dir, f"{sub_id}_GT.nii.gz"))
            nib.save(nib.Nifti1Image(recon_mask.cpu().numpy().squeeze(), affine),
                     os.path.join(run_dir, f"{sub_id}_Pred.nii.gz"))

    # Produce the final reporting artifacts
    pd.DataFrame(results).to_csv(os.path.join(run_dir, f"report_{run_name}.csv"), index=False)
    
    print(f"\n✅ Diagnostics Complete. Files securely saved to: {run_dir}")


if __name__ == "__main__":
    run_vae_evaluation()