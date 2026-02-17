import os
import sys
import csv
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.stats import pearsonr

# --- 1. Dynamic Path Adjustment & Model Import ---
current_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(current_dir, 'utils', 'models')
if models_path not in sys.path:
    sys.path.append(models_path)

try:
    from vae import VAE3D

    print("✅ Successfully imported VAE3D architecture.")
except ImportError as e:
    print(f"❌ CRITICAL: Could not find 'vae.py' or 'VAE3D' class. Error: {e}")
    exit()

# --- 2. Path Configuration ---
BASE_PATH = r"C:\Users\Lab\OneDrive\Desktop\SPQA"
FOLD_SOURCE_DIR = os.path.join(BASE_PATH, "folds")
LOG_BASE_DIR = os.path.join(BASE_PATH, "logs", "validation_logs")
MASTER_CSV_PATH = os.path.join(LOG_BASE_DIR, "master_validation_metrics.csv")
MODEL_PATH = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\vae_checkpoints\vae_epoch_460.pth"

os.makedirs(LOG_BASE_DIR, exist_ok=True)

# --- 3. Model Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE3D().to(device)

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Loaded VAE3D Epoch 460 on {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ Model Load Error: {e}")
    exit()


# --- 4. Helper Functions ---
def dice_score(mask1, mask2):
    mask1, mask2 = (mask1 > 0.5).astype(bool), (mask2 > 0.5).astype(bool)
    if not np.any(mask1) and not np.any(mask2): return 1.0
    intersection = np.logical_and(mask1, mask2).sum()
    return (2. * intersection) / (mask1.sum() + mask2.sum() + 1e-8)


def corrupt_mask(mask, iterations=2):
    if np.random.rand() > 0.5:
        return binary_dilation(mask, iterations=iterations).astype(mask.dtype)
    return binary_erosion(mask, iterations=iterations).astype(mask.dtype)


def reconstruct_model_predict(image_data, corrupt_mask_data):
    """
    FIXED: Concatenates Image and Corrupt Mask to create a 2-channel input.
    Expected Input Shape: [1, 2, Z, Y, X]
    """
    # 1. Normalize image if necessary (assumes VAE expects 0-1 range)
    img_tensor = torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.from_numpy(corrupt_mask_data).float().unsqueeze(0).unsqueeze(0)

    # 2. Concatenate along the channel dimension (dim=1)
    # Resulting shape: [1, 2, 256, 256, 256]
    input_tensor = torch.cat((img_tensor, mask_tensor), dim=1).to(device)

    with torch.no_grad():
        # VAE3D returns (recon, mu, logvar)
        recon_output, _, _ = model(input_tensor)
        recon_probs = torch.sigmoid(recon_output)

    return recon_probs.squeeze().cpu().numpy()


def save_comparison_plot(image_data, gt, corrupt, recon, save_path, dp_id, fold_name):
    mid_slice = gt.shape[2] // 2
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    slices = [image_data[:, :, mid_slice], gt[:, :, mid_slice],
              corrupt[:, :, mid_slice], recon[:, :, mid_slice]]
    titles = ['Scan Slice', 'GT (truth)', 'Corrupted Mask', 'VAE Recon']
    for ax, s, t in zip(axes, slices, titles):
        ax.imshow(s, cmap='gray')
        ax.set_title(f"{t}\n{fold_name} | {dp_id}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --- 5. Main Processing ---
try:
    with open(MASTER_CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fold', 'Datapoint_ID', 'True_Dice', 'Estimated_Dice', 'Voxel_Correlation'])
except PermissionError:
    print(f"❌ PERMISSION DENIED: Close {MASTER_CSV_PATH} and try again.")
    exit()

missing_report = []

for i in range(5):
    fold_name = f"fold_{i}"
    fold_root = os.path.join(FOLD_SOURCE_DIR, fold_name)

    if not os.path.exists(fold_root):
        missing_report.append(f"{fold_name}: Folder missing")
        continue

    id_file_path = os.path.join(fold_root, fold_name, "validation_ids.txt")
    if not os.path.exists(id_file_path):
        missing_report.append(f"{fold_name}: validation_ids.txt missing")
        continue

    with open(id_file_path, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]

    fold_log_dir = os.path.join(LOG_BASE_DIR, fold_name)
    os.makedirs(fold_log_dir, exist_ok=True)

    print(f"\n--- Processing {fold_name} ---")

    for dp_id in val_ids:
        mask_path = os.path.join(fold_root, "pseudo_labels", dp_id, "truth.nii.gz")
        image_path = os.path.join(fold_root, "pseudo_labels", dp_id, "image.nii.gz")

        if not os.path.exists(mask_path) or not os.path.exists(image_path):
            missing_report.append(f"{fold_name} | {dp_id}: truth or image missing")
            continue

        try:
            gt_nifti = nib.load(mask_path)
            gt_data = gt_nifti.get_fdata()
            img_data = nib.load(image_path).get_fdata()

            corrupt_data = corrupt_mask(gt_data)

            # Pass both image and mask to the prediction function
            recon_data = reconstruct_model_predict(img_data, corrupt_data)

            t_dice = dice_score(gt_data, corrupt_data)
            e_dice = dice_score(gt_data, recon_data)
            v_corr, _ = pearsonr(gt_data.flatten(), recon_data.flatten())

            dp_folder = os.path.join(fold_log_dir, dp_id)
            os.makedirs(dp_folder, exist_ok=True)

            nib.save(nib.Nifti1Image(recon_data, gt_nifti.affine),
                     os.path.join(dp_folder, f"{dp_id}_recon_attempt.nii.gz"))

            save_comparison_plot(img_data, gt_data, corrupt_data, recon_data,
                                 os.path.join(dp_folder, f"{dp_id}_comparison_plot.png"), dp_id, fold_name)

            with open(MASTER_CSV_PATH, mode='a', newline='') as f:
                csv.writer(f).writerow([fold_name, dp_id, t_dice, e_dice, v_corr])

            print(f"  ✅ {dp_id} | Corr: {v_corr:.4f}")

        except Exception as e:
            print(f"  ❌ {dp_id} Error: {e}")

if missing_report:
    print("\n--- Missing File Report ---")
    for msg in missing_report: print(f" - {msg}")

print(f"\n🚀 Complete! Review logs in: {LOG_BASE_DIR}")