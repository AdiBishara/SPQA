import os
import sys
import glob
import re
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.transforms import AsDiscrete
from utils.config import load_config
from utils.models.unet_dropout import UNet


# --- HELPER: Find Best Checkpoint ---
def get_latest_checkpoint(save_dir, prefix="unet3d_epoch_"):
    """Finds the checkpoint with the highest epoch number."""
    checkpoints = glob.glob(os.path.join(save_dir, f"{prefix}*.pth"))
    if not checkpoints:
        return None
    latest_ckpt = max(checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    return latest_ckpt


def run_evaluation():
    # 1. LOAD CONFIG & SETUP
    config_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- Starting Evaluation & Reconstruction on {device} ---")

    # 2. OUTPUT FOLDERS
    # We create a clean folder for your thesis/paper results
    output_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\results_evaluation"
    nifti_dir = os.path.join(output_dir, "nifti_predictions")
    os.makedirs(nifti_dir, exist_ok=True)

    # 3. LOAD U-NET MODEL (The 'Artist')
    print("Loading Segmentation Model...")
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['n_classes'],
        channels=config['model']['channels'],
        strides=config['model']['strides'],
        dropout=0.0,  # Disable dropout for deterministic validation
        spatial_dims=3
    ).to(device)

    # Find and load weights
    ckpt_dir = config['Train']['save_dir']
    ckpt_path = get_latest_checkpoint(ckpt_dir)

    if ckpt_path:
        print(f"✅ Loaded weights: {os.path.basename(ckpt_path)}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
    else:
        print(f"❌ Error: No checkpoint found in {ckpt_dir}")
        return

    # 4. SETUP METRICS (SingleStrip Paper Standards)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    # HD95: 95th Percentile Hausdorff Distance (measures boundary errors)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    post_pred = AsDiscrete(threshold=0.5)

    results = []

    # 5. GET TEST DATA
    with open(config['Data']['test_ids'], 'r') as f:
        test_ids = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Processing {len(test_ids)} subjects...")

    # 6. INFERENCE LOOP
    for subject_id in test_ids:
        print(f"Processing: {subject_id}...", end="")

        # Paths
        base_path = os.path.join(config['Data']['raw_data_root'], subject_id)
        img_path = os.path.join(base_path, "image.nii.gz")
        mask_path = os.path.join(base_path, "mask.nii.gz")

        # Handle .nii extension variations
        if not os.path.exists(img_path): img_path = img_path.replace(".nii.gz", ".nii")
        if not os.path.exists(mask_path): mask_path = mask_path.replace(".nii.gz", ".nii")

        # Load NIfTI (We need the affine/header for saving)
        nifti_img = nib.load(img_path)
        nifti_mask = nib.load(mask_path)

        img_data = nifti_img.get_fdata().astype(np.float32)
        mask_data = nifti_mask.get_fdata().astype(np.float32)

        # Normalize Image (Same as training)
        if np.max(img_data) > 0:
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        # --- RESIZE LOGIC (Match Training 256x256x256) ---
        d, h, w = img_data.shape
        target_size = (256, 256, 256)

        # Create Canvas (Padded)
        canvas = np.zeros(target_size, dtype=np.float32)

        # Calculate Offsets to Center the Brain
        z_off = max(0, (target_size[0] - d) // 2)
        y_off = max(0, (target_size[1] - h) // 2)
        x_off = max(0, (target_size[2] - w) // 2)

        z_end = min(target_size[0], z_off + d)
        y_end = min(target_size[1], y_off + h)
        x_end = min(target_size[2], x_off + w)

        # Paste Image into Canvas
        canvas[z_off:z_end, y_off:y_end, x_off:x_end] = img_data[:z_end - z_off, :y_end - y_off, :x_end - x_off]

        # Prepare Tensors
        input_tensor = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0).to(device)

        # Pad Ground Truth to match prediction for metric calculation
        gt_canvas = np.zeros(target_size, dtype=np.float32)
        gt_canvas[z_off:z_end, y_off:y_end, x_off:x_end] = mask_data[:z_end - z_off, :y_end - y_off, :x_end - x_off]
        gt_tensor_padded = torch.from_numpy(gt_canvas).unsqueeze(0).unsqueeze(0).to(device)

        # --- INFERENCE ---
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = model(input_tensor)
                pred_prob = torch.sigmoid(output)
                pred_mask = post_pred(pred_prob)

            # --- CALCULATE METRICS ---
            # 1. Dice
            dice_metric(y_pred=pred_mask, y=gt_tensor_padded)
            dice_score = dice_metric.aggregate().item()

            # 2. Jaccard (IoU)
            iou_metric(y_pred=pred_mask, y=gt_tensor_padded)
            iou_score = iou_metric.aggregate().item()

            # 3. Hausdorff Distance (HD95)
            # Note: HD95 can be finicky if mask is empty. We handle errors inside MONAI usually.
            hd95_metric(y_pred=pred_mask, y=gt_tensor_padded)
            hd95_score = hd95_metric.aggregate().item()

            # Reset metrics for next loop
            dice_metric.reset()
            iou_metric.reset()
            hd95_metric.reset()

        # --- SAVE 3D RECONSTRUCTION ---
        # 1. Extract the brain from the padded canvas (Reverse padding)
        pred_numpy = pred_mask.cpu().numpy().squeeze().astype(np.uint8)
        final_pred = pred_numpy[z_off:z_end, y_off:y_end, x_off:x_end]

        # 2. Create NIfTI with ORIGINAL Header/Affine
        # This ensures the new file lines up perfectly with the original scan
        pred_nifti = nib.Nifti1Image(final_pred, nifti_img.affine, nifti_img.header)

        # 3. Save
        save_name = f"{subject_id}_pred.nii.gz"
        nib.save(pred_nifti, os.path.join(nifti_dir, save_name))

        # Log to Console
        print(f" Dice: {dice_score:.4f} | IoU: {iou_score:.4f} | HD95: {hd95_score:.2f}")

        results.append({
            "Subject_ID": subject_id,
            "Dice": dice_score,
            "Jaccard_IoU": iou_score,
            "Hausdorff95": hd95_score
        })

    # 7. SAVE CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "singlestrip_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 30)
    print(" EVALUATION COMPLETE")
    print(f" Mean Dice: {df['Dice'].mean():.4f}")
    print(f" Mean HD95: {df['Hausdorff95'].mean():.4f}")
    print(f" CSV Saved to: {csv_path}")
    print(f" NIfTI Reconstructions in: {nifti_dir}")
    print("=" * 30)


if __name__ == "__main__":
    run_evaluation()