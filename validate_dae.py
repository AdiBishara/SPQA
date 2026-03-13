import os
import sys
import argparse
import torch
import numpy as np
import nibabel as nib
import matplotlib
import glob

# Ensure that we don't try to open GUI windows when running these scripts
# on a remote server or in bulk processing jobs.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from utils.models.unet_dae import UNetDAE


# --- Core Helper Functions ---

def dice_score(mask1, mask2):
    """Computes basic discrete Dice score, safely handling zero-division for empty masks."""
    m1 = (mask1 > 0.5).astype(bool)
    m2 = (mask2 > 0.5).astype(bool)
    
    # If both masks are entirely blank, they perfectly match!
    if not np.any(m1) and not np.any(m2): 
        return 1.0
        
    return (2. * np.logical_and(m1, m2).sum()) / (m1.sum() + m2.sum() + 1e-8)


def reconstruct_model_predict(model, device, image_data, pseudo_label_data):
    """Passes padded 3D subject through DAE, discarding dummy mu/logvar to yield reconstruction."""
    img = torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0)
    msk = torch.from_numpy(pseudo_label_data).float().unsqueeze(0).unsqueeze(0)
    
    # The network expects the raw MRI image and the mask to be stacked as channels
    input_tensor = torch.cat((img, msk), dim=1).to(device)
    
    with torch.no_grad():
        recon, _, _ = model(input_tensor)
        # Squeeze down to the raw 3D volume
        return torch.sigmoid(recon).squeeze().cpu().numpy()


def save_comparison_plot(image_data, gt, pseudo, recon, save_path, dp_id, fold_name, cp_name):
    """Generates a 4-panel visual report card (MRI, Corrupt, Diff, Pred) for subject evaluation."""
    mid = gt.shape[2] // 2
    img_s = image_data[:, :, mid]
    gt_s = gt[:, :, mid]
    ps_s = pseudo[:, :, mid]
    recon_s = (recon[:, :, mid] > 0.5).astype(float)
    diff_s = np.abs(gt_s - ps_s)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    titles = ['Ground Truth', 'Corrupted', 'Difference', f'Prediction\n({cp_name})']

    for ax, title in zip(axes, titles):
        ax.imshow(img_s, cmap='gray')
        ax.set_title(f"{title}\n{fold_name} | {dp_id}")
        ax.axis('off')

    # Apply colored overlays over the brain MRI
    if np.any(gt_s): 
        axes[0].imshow(np.ma.masked_where(gt_s < 0.5, gt_s), cmap='Greens', alpha=0.45, vmin=0, vmax=1, interpolation='nearest')
    if np.any(ps_s): 
        axes[1].imshow(np.ma.masked_where(ps_s < 0.5, ps_s), cmap='autumn', alpha=0.45, vmin=0, vmax=1, interpolation='nearest')
    if np.any(diff_s): 
        axes[2].imshow(np.ma.masked_where(diff_s < 0.5, diff_s), cmap='Wistia', alpha=0.55, vmin=0, vmax=1, interpolation='nearest')
    if np.any(recon_s): 
        axes[3].imshow(np.ma.masked_where(recon_s < 0.5, recon_s), cmap='Reds', alpha=0.45, vmin=0, vmax=1, interpolation='nearest')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --- Validation Routine ---

def validate_checkpoint(checkpoint_path, base_log_dir, fold_source_dir):
    """Executes full 5-fold validation on a DAE checkpoint to verify generalization."""
    cp_name = os.path.basename(checkpoint_path).replace('.pth', '')
    run_log_dir = os.path.join(base_log_dir, "dae", cp_name)
    os.makedirs(run_log_dir, exist_ok=True)
    
    master_csv_path = os.path.join(run_log_dir, "validation_metrics.csv")
    summary_txt_path = os.path.join(run_log_dir, "global_summary.txt")

    print(f"\n{'='*50}")
    print(f"STARTING VALIDATION: {cp_name} (dae)")
    print(f"{'='*50}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDAE(in_channels=2, out_channels=1, spatial_dims=3).to(device)

    # 1. Boot up the DAE architecture with the requested weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded successfully.")
    except Exception as e:
        print(f"Load Error: {e}")
        return

    all_subject_data = []

    # 2. Iterate through Fold 0 to Fold 4
    for i in range(5):
        fold_name = f"fold_{i}"
        pseudo_labels_dir = os.path.join(fold_source_dir, fold_name, "pseudo_labels")
        if not os.path.exists(pseudo_labels_dir): 
            continue

        all_ids = [d for d in os.listdir(pseudo_labels_dir) if os.path.isdir(os.path.join(pseudo_labels_dir, d))]
        print(f"--- Fold {i}: Auditing {len(all_ids)} subjects ---")
        
        fold_log_dir = os.path.join(run_log_dir, fold_name)
        os.makedirs(fold_log_dir, exist_ok=True)

        # 3. Process every subject inside this fold
        for dp_id in all_ids:
            dp_path = os.path.join(pseudo_labels_dir, dp_id)
            t_path = os.path.join(dp_path, "truth.nii.gz")
            p_path = os.path.join(dp_path, "pseudo_label.nii.gz")
            i_path = os.path.join(dp_path, "image.nii.gz")

            # Skip if any data stream is missing
            if not all(os.path.exists(p) for p in [t_path, p_path, i_path]): 
                continue

            try:
                # Load the NIFTI volumes into memory
                gt_d = nib.load(t_path).get_fdata()
                ps_d = nib.load(p_path).get_fdata()
                im_d = nib.load(i_path).get_fdata()

                # --- Normalization Strategy ---
                # We calculate standard deviation ONLY over the foreground brain voxels
                # avoiding the dark background which would compress the meaningful contrast
                img = im_d.astype(np.float32)
                foreground_mask = img > 0
                foreground_voxels = img[foreground_mask]
                
                if len(foreground_voxels) > 0:
                    mean, std = foreground_voxels.mean(), foreground_voxels.std()
                    img = (img - mean) / (std + 1e-8)
                    img = np.clip(img, -3.0, 3.0)
                    img[~foreground_mask] = -3.0
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                else: 
                    img = np.zeros_like(img)

                # --- 3D Network Padding ---
                # The UNet logic inside DAE requires volumes sized in multiples of 16/32
                target_size = (256, 256, 256)
                d, h, w = img.shape
                td, th, tw = target_size
                pad_d, pad_h, pad_w = max(0, td - d), max(0, th - h), max(0, tw - w)
                
                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    pd1, pd2 = pad_d // 2, pad_d - (pad_d // 2)
                    ph1, ph2 = pad_h // 2, pad_h - (pad_h // 2)
                    pw1, pw2 = pad_w // 2, pad_w - (pad_w // 2)
                    img = np.pad(img, ((pd1, pd2), (ph1, ph2), (pw1, pw2)), mode='constant')
                    ps_d_padded = np.pad(ps_d, ((pd1, pd2), (ph1, ph2), (pw1, pw2)), mode='constant')
                else: 
                    ps_d_padded = ps_d.copy()

                # Center crop if the image is somehow larger than the target max box
                if d > td or h > th or w > tw:
                    z, y, x = (d - td) // 2, (h - th) // 2, (w - tw) // 2
                    img = img[z:z + td, y:y + th, x:x + tw]
                    ps_d_padded = ps_d_padded[z:z + td, y:y + th, x:x + tw]

                # --- Network Inference ---
                recon_padded = reconstruct_model_predict(model, device, img, ps_d_padded)

                # Reverse the padding to get the prediction back into the native NIFTI shape
                recon = np.zeros_like(ps_d)
                od, oh, ow = ps_d.shape
                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    recon = recon_padded[pd1:pd1 + od, ph1:ph1 + oh, pw1:pw1 + ow]
                elif od > td or oh > th or ow > tw:
                    z, y, x = (od - td) // 2, (oh - th) // 2, (ow - tw) // 2
                    recon[z:z + td, y:y + th, x:x + tw] = recon_padded
                else: 
                    recon = recon_padded

                # --- Metric Gathering ---
                # Actual Dice: What the pseudo label actually scored against the human annotator
                act_d = dice_score(gt_d, ps_d)
                # Estimated Dice: What our AI model thinks the pseudo label scored
                est_d = dice_score(ps_d, recon)
                
                all_subject_data.append({'Fold': fold_name, 'ID': dp_id, 'Actual': act_d, 'Estimated': est_d})

                # Export a visualization for human review
                dp_folder = os.path.join(fold_log_dir, dp_id)
                os.makedirs(dp_folder, exist_ok=True)
                save_comparison_plot(im_d, gt_d, ps_d, recon, os.path.join(dp_folder, "comparison.png"), dp_id, fold_name, cp_name)
                
            except Exception as e:
                print(f"Fail {dp_id}: {e}")

    # 4. Generate Final Reports
    df = pd.DataFrame(all_subject_data)
    df.to_csv(master_csv_path, index=False)

    if not df.empty:
        # Build the final text summary table summarizing correlation across all folds
        global_report = ["=" * 100, f"SPQA SCIENTIFIC VALIDATION SUMMARY | {cp_name}", "=" * 100]
        header = f"{'Fold':<10} | {'N':<4} | {'Pearson r':<12} | {'MAE':<10} | {'Actual Dice':<20} | {'Estimated Dice':<20}"
        global_report.append(header); global_report.append("-" * 100)

        fold_r_vals, fold_mae_vals = [], []
        for fold in sorted(df['Fold'].unique()):
            f_df = df[df['Fold'] == fold]
            n = len(f_df)
            if n > 1:
                f_act, f_est = f_df['Actual'].values, f_df['Estimated'].values
                r_val, p_val = pearsonr(f_act, f_est)
                mae_val = mean_absolute_error(f_act, f_est)
                act_mu, act_std = np.mean(f_act), np.std(f_act)
                est_mu, est_std = np.mean(f_est), np.std(f_est)
                fold_r_vals.append(r_val); fold_mae_vals.append(mae_val)
                global_report.append(f"{fold:<10} | {n:<4} | {r_val:0.4f}       | {mae_val:0.4f}     | {act_mu:0.3f} +/- {act_std:0.3f}       | {est_mu:0.3f} +/- {est_std:0.3f}")

        # Compute overarching Global Pearson Correlation Score
        all_act, all_est = df['Actual'].values, df['Estimated'].values
        g_r, g_p = pearsonr(all_act, all_est)
        g_mae = mean_absolute_error(all_act, all_est)

        global_report.append("-" * 100)
        global_report.append(f"{'OVERALL':<10} | {len(df):<4} | {g_r:0.3f} +/- {np.std(fold_r_vals):0.3f} | {g_mae:0.3f} +/- {np.std(fold_mae_vals):0.3f} | {np.mean(all_act):0.3f} +/- {np.std(all_act):0.3f}       | {np.mean(all_est):0.3f} +/- {np.std(all_est):0.3f}")
        global_report.append("-" * 100); global_report.append("")

        # Add helpful English interpretation of the math for quick reading
        if g_r >= 0.7: indication = "STRONG: Excellent generalization expected."
        elif g_r >= 0.5: indication = "MODERATE: Showing good structural trends."
        elif g_r >= 0.3: indication = "WEAK: Further tuning required. Weak predictive power."
        else: indication = "POOR: Model isn't capturing variations well. It's wildly off."

        global_report.extend([f"Run Checked:      {cp_name}", f"Total Samples:    {len(df)}", f"Pearson r:        {g_r:.4f}", f"MAE:              {g_mae:.4f}", "-" * 50, f"Indication:       {indication}", "=" * 50])

        summary_text = "\n".join(global_report)
        print(summary_text)
        with open(summary_txt_path, 'w') as f: 
            f.write(summary_text)


def main():
    # Setup working tree dynamics
    base_path = os.path.dirname(os.path.abspath(__file__))
    fold_source_dir = os.path.join(base_path, "folds")
    
    # Checkpoints to test (add specific DAE checkpoints here as needed)
    dae_cps = [
        r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\best_cps\dae_run_2_best_cps\dae_epoch_421.pth",
        r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\best_cps\dae_run_2_best_cps\dae_epoch_420.pth",
        r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\best_cps\dae_run_2_best_cps\dae_epoch_427.pth"
    ]
    
    base_output_dir = os.path.join(base_path, "logs", "final_validation_results")
    if not os.path.exists(base_output_dir): 
        os.makedirs(base_output_dir)

    print("--- STARTING DAE BATCH VALIDATION ---\n")
    for cp in dae_cps:
        if os.path.exists(cp): 
            validate_checkpoint(cp, base_output_dir, fold_source_dir)
        else: 
            print(f"File not found, skipping: {cp}")

    print("\nBatch validation completed successfully!")
    print(f"Results are saved in: {base_output_dir}")

if __name__ == "__main__":
    main()
