import os
import sys
import csv
import torch
import shutil
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# --- 1. DYNAMIC PATH ADJUSTMENT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(current_dir, 'utils', 'models')
if models_path not in sys.path: sys.path.append(models_path)

try:
    from vae import VAE3D

    print("Successfully imported VAE3D architecture.")
except ImportError as e:
    print(f"CRITICAL: Could not find 'vae.py'. Error: {e}");
    exit()

# --- 2. PATH CONFIGURATION ---
BASE_PATH = r"C:\Users\Lab\OneDrive\Desktop\SPQA"
FOLD_SOURCE_DIR = os.path.join(BASE_PATH, "folds")
LOG_BASE_DIR = os.path.join(BASE_PATH, "logs", "validation_logs")
MASTER_CSV_PATH = os.path.join(LOG_BASE_DIR, "master_validation_metrics.csv")
SUMMARY_TXT_PATH = os.path.join(LOG_BASE_DIR, "global_validation_summary.txt")

# Audit Target
MODEL_PATH = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\best_cps\run_22_best_cps\vae_epoch_733.pth"


# --- 3. THE CLEANER ---
def clean_validation_dir(log_dir):
    """
    Deletes everything inside the validation log directory to prevent
    mixing results from different checkpoints.
    """
    if os.path.exists(log_dir):
        print(f"🧹 Cleaning previous results in: {log_dir}")
        for item in os.listdir(log_dir):
            item_path = os.path.join(log_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"   [Error cleaning {item}]: {e}")
    os.makedirs(log_dir, exist_ok=True)


clean_validation_dir(LOG_BASE_DIR)

# --- 4. MODEL INITIALIZATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE3D(latent_dim=4096).to(device)
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded Checkpoint: {os.path.basename(MODEL_PATH)}")
except Exception as e:
    print(f"Load Error: {e}"); exit()


# --- 5. HELPERS ---
def dice_score(mask1, mask2):
    m1, m2 = (mask1 > 0.5).astype(bool), (mask2 > 0.5).astype(bool)
    if not np.any(m1) and not np.any(m2): return 1.0
    return (2. * np.logical_and(m1, m2).sum()) / (m1.sum() + m2.sum() + 1e-8)


def reconstruct_model_predict(image_data, pseudo_label_data):
    img = torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0)
    msk = torch.from_numpy(pseudo_label_data).float().unsqueeze(0).unsqueeze(0)
    input_tensor = torch.cat((img, msk), dim=1).to(device)
    with torch.no_grad():
        recon, _, _ = model(input_tensor)
        return torch.sigmoid(recon).squeeze().cpu().numpy()


def save_comparison_plot(image_data, gt, pseudo, recon, save_path, dp_id, fold_name):
    mid = gt.shape[2] // 2
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    data = [image_data[:, :, mid], gt[:, :, mid], pseudo[:, :, mid], recon[:, :, mid]]
    titles = ['Scan', 'GT', 'Pseudo', 'VAE Recon']
    for ax, d, t in zip(axes, data, titles):
        ax.imshow(d, cmap='gray');
        ax.set_title(f"{t}\n{fold_name} | {dp_id}");
        ax.axis('off')
    plt.tight_layout();
    plt.savefig(save_path);
    plt.close()


# --- 6. MAIN PROCESSING LOOP ---
all_subject_data = []

for i in range(5):
    fold_name = f"fold_{i}"
    pseudo_labels_dir = os.path.join(FOLD_SOURCE_DIR, fold_name, "pseudo_labels")
    if not os.path.exists(pseudo_labels_dir): continue

    all_ids = [d for d in os.listdir(pseudo_labels_dir) if os.path.isdir(os.path.join(pseudo_labels_dir, d))]
    print(f"\n--- Fold {i}: Auditing {len(all_ids)} subjects ---")
    fold_log_dir = os.path.join(LOG_BASE_DIR, fold_name);
    os.makedirs(fold_log_dir, exist_ok=True)

    for dp_id in all_ids:
        dp_path = os.path.join(pseudo_labels_dir, dp_id)
        t_path, p_path, i_path = os.path.join(dp_path, "truth.nii.gz"), os.path.join(dp_path,
                                                                                     "pseudo_label.nii.gz"), os.path.join(
            dp_path, "image.nii.gz")
        if not all(os.path.exists(p) for p in [t_path, p_path, i_path]): continue

        try:
            gt_d, ps_d, im_d = nib.load(t_path).get_fdata(), nib.load(p_path).get_fdata(), nib.load(i_path).get_fdata()
            recon = reconstruct_model_predict(im_d, ps_d)
            act_d, est_d = dice_score(gt_d, ps_d), dice_score(ps_d, recon)

            all_subject_data.append({'Fold': fold_name, 'ID': dp_id, 'Actual': act_d, 'Estimated': est_d})

            dp_folder = os.path.join(fold_log_dir, dp_id);
            os.makedirs(dp_folder, exist_ok=True)
            save_comparison_plot(im_d, gt_d, ps_d, recon, os.path.join(dp_folder, f"comparison.png"), dp_id, fold_name)
            print(f"   Done: {dp_id} | Actual: {act_d:.3f} | Est: {est_d:.3f}", end="\r")
        except Exception as e:
            print(f"\nFail {dp_id}: {e}")

# --- 7. STATISTICS ANALYSIS ---
df = pd.DataFrame(all_subject_data)
df.to_csv(MASTER_CSV_PATH, index=False)

if not df.empty:
    global_report = ["=" * 135, "SPQA SCIENTIFIC VALIDATION SUMMARY (μ ± σ)", "=" * 135]
    global_report.append(f"Checkpoint: {os.path.basename(MODEL_PATH)}")
    global_report.append("-" * 135)
    header = (f"{'Fold':<10} | {'N':<4} | {'Pearson r (μ ± σ)':<22} | {'MAE (μ ± σ)':<18} | "
              f"{'Actual Dice (μ ± σ)':<25} | {'Estimated Dice (μ ± σ)':<25}")
    global_report.append(header);
    global_report.append("-" * 135)

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
            fold_r_vals.append(r_val);
            fold_mae_vals.append(mae_val)

            global_report.append(f"{fold:<10} | {n:<4} | {r_val:0.4f}               | {mae_val:0.4f}           | "
                                 f"{act_mu:0.3f} ± {act_std:0.3f}       | {est_mu:0.3f} ± {est_std:0.3f}")

            with open(os.path.join(LOG_BASE_DIR, fold, f"{fold}_stats.txt"), 'w') as ff:
                ff.write(
                    f"Pearson r: {r_val:0.4f}\nMAE: {mae_val:0.4f}\nActual: {act_mu:0.4f}±{act_std:0.4f}\nEst: {est_mu:0.4f}±{est_std:0.4f}")

    all_act, all_est = df['Actual'].values, df['Estimated'].values
    g_r, g_p = pearsonr(all_act, all_est)
    g_mae = mean_absolute_error(all_act, all_est)

    global_report.append("-" * 135)
    global_report.append(f"{'OVERALL':<10} | {len(df):<4} | {g_r:0.3f} ± {np.std(fold_r_vals):0.3f}       | "
                         f"{g_mae:0.3f} ± {np.std(fold_mae_vals):0.3f}   | "
                         f"{np.mean(all_act):0.3f} ± {np.std(all_act):0.3f}       | "
                         f"{np.mean(all_est):0.3f} ± {np.std(all_est):0.3f}")
    global_report.append("-" * 135)

    summary_text = "\n".join(global_report)
    print("\n" + summary_text)
    with open(SUMMARY_TXT_PATH, 'w') as f:
        f.write(summary_text)

print(f"\nAudit complete. Logs cleaned and results stored in: {LOG_BASE_DIR}")