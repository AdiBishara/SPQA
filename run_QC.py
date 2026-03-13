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


# --- Utilities ---

def get_latest_checkpoint(save_dir, prefix="unet3d_epoch_"):
    """Scans the directory for the checkpoint file with the highest epoch number."""
    checkpoints = glob.glob(os.path.join(save_dir, f"{prefix}*.pth"))
    if not checkpoints:
        return None
    
    # Extract the epoch number using RegEx and pick the highest one
    latest_ckpt = max(checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    return latest_ckpt


# --- Main Evaluation Routine ---

def run_evaluation():
    """Evaluates UNet segmentation on test subjects, generating metrics and NIfTI predictions."""
    # 1. Boot up configurations and hardware
    config_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- Starting Evaluation & Reconstruction on {device} ---")

    # 2. Prepare the output stage
    # This directory holds everything needed for the thesis/paper results
    output_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\results_evaluation"
    nifti_dir = os.path.join(output_dir, "nifti_predictions")
    os.makedirs(nifti_dir, exist_ok=True)

    # 3. Initialize the UNet 'Artist'
    print("Loading Segmentation Model Architecture...")
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['n_classes'],
        channels=config['model']['channels'],
        strides=config['model']['strides'],
        dropout=0.0,  # We strictly disable dropout for deterministic validation
        spatial_dims=3
    ).to(device)

    # Hunt down the best weights and load them into the model
    ckpt_dir = config['Train']['save_dir']
    ckpt_path = get_latest_checkpoint(ckpt_dir)

    if ckpt_path:
        print(f"✅ Loaded weights: {os.path.basename(ckpt_path)}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
    else:
        print(f"❌ Error: No valid checkpoint found in {ckpt_dir}")
        return

    # 4. Initialize Clinical Standard Metrics (MONAI)
    # These represent the exact benchmarks accepted in top medical imaging papers
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    
    # HD95: 95th Percentile Hausdorff Distance (measures extreme boundary errors)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    post_pred = AsDiscrete(threshold=0.5)
    results = []

    # 5. Fetch the held-out Test Data IDs
    with open(config['Data']['test_ids'], 'r') as f:
        test_ids = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Processing {len(test_ids)} subjects...")

    # 6. Execute the Inference Loop across every subject
    for subject_id in test_ids:
        print(f"Processing: {subject_id}...", end="")

        # Resolve paths gracefully, accounting for different NIfTI extensions
        base_path = os.path.join(config['Data']['raw_data_root'], subject_id)
        img_path = os.path.join(base_path, "image.nii.gz")
        mask_path = os.path.join(base_path, "mask.nii.gz")

        if not os.path.exists(img_path): img_path = img_path.replace(".nii.gz", ".nii")
        if not os.path.exists(mask_path): mask_path = mask_path.replace(".nii.gz", ".nii")

        # Load NIfTI — we keep the native affine/header metadata so we can 
        # save our AI prediction perfectly aligned in medical software (like ITK-SNAP)
        nifti_img = nib.load(img_path)
        nifti_mask = nib.load(mask_path)

        img_data = nifti_img.get_fdata().astype(np.float32)
        mask_data = nifti_mask.get_fdata().astype(np.float32)

        # Apply standardized intensity normalization identical to training
        if np.max(img_data) > 0:
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        # --- Padding Logic ---
        # The UNet requires rigid power-of-2 dimensions (256x256x256)
        d, h, w = img_data.shape
        target_size = (256, 256, 256)

        # Create a black padded canvas
        canvas = np.zeros(target_size, dtype=np.float32)

        # Calculate exactly how to center the patient's brain within this canvas
        z_off = max(0, (target_size[0] - d) // 2)
        y_off = max(0, (target_size[1] - h) // 2)
        x_off = max(0, (target_size[2] - w) // 2)

        z_end = min(target_size[0], z_off + d)
        y_end = min(target_size[1], y_off + h)
        x_end = min(target_size[2], x_off + w)

        # Paste the patient's brain directly into the center
        canvas[z_off:z_end, y_off:y_end, x_off:x_end] = img_data[:z_end - z_off, :y_end - y_off, :x_end - x_off]

        # Convert to a PyTorch Batch (B, C, D, H, W)
        input_tensor = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0).to(device)

        # Repeat padding for the Ground Truth mask to ensure fair metric comparison
        gt_canvas = np.zeros(target_size, dtype=np.float32)
        gt_canvas[z_off:z_end, y_off:y_end, x_off:x_end] = mask_data[:z_end - z_off, :y_end - y_off, :x_end - x_off]
        gt_tensor_padded = torch.from_numpy(gt_canvas).unsqueeze(0).unsqueeze(0).to(device)

        # --- Network Inference ---
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                # Extract the prediction and squash it to probabilities
                output = model(input_tensor)
                pred_prob = torch.sigmoid(output)
                
                # Snap to a hard binary mask (1 or 0)
                pred_mask = post_pred(pred_prob)

            # --- Clinical Metric Calculation ---
            # 1. Dice Score: Overall spatial overlap
            dice_metric(y_pred=pred_mask, y=gt_tensor_padded)
            dice_score = dice_metric.aggregate().item()

            # 2. Jaccard (IoU): Intersection over Union (harsher than Dice)
            iou_metric(y_pred=pred_mask, y=gt_tensor_padded)
            iou_score = iou_metric.aggregate().item()

            # 3. Hausdorff Distance (HD95): Evaluates structural boundary errors
            # (Note: HD95 can crash if the mask is totally empty, MONAI generally catches this)
            hd95_metric(y_pred=pred_mask, y=gt_tensor_padded)
            hd95_score = hd95_metric.aggregate().item()

            # Flush the running metric buffers for the next patient
            dice_metric.reset()
            iou_metric.reset()
            hd95_metric.reset()

        # --- Exporting 3D NIfTI Reconstructions ---
        # 1. Slice off the temporary zero-padding to return to the patient's native shape
        pred_numpy = pred_mask.cpu().numpy().squeeze().astype(np.uint8)
        final_pred = pred_numpy[z_off:z_end, y_off:y_end, x_off:x_end]

        # 2. Inject the prediction back into NIfTI using the original medical Header/Affine.
        # This is critical so the file isn't scrambled/rotated when loaded by doctors.
        pred_nifti = nib.Nifti1Image(final_pred, nifti_img.affine, nifti_img.header)

        # 3. Commit to disk
        save_name = f"{subject_id}_pred.nii.gz"
        nib.save(pred_nifti, os.path.join(nifti_dir, save_name))

        # Real-time Terminal Feedback
        print(f" Dice: {dice_score:.4f} | IoU: {iou_score:.4f} | HD95: {hd95_score:.2f}")

        # Accumulate metrics
        results.append({
            "Subject_ID": subject_id,
            "Dice": dice_score,
            "Jaccard_IoU": iou_score,
            "Hausdorff95": hd95_score
        })

    # 7. Final Output Report Generation (CSV)
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "singlestrip_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 50)
    print(" EVALUATION COMPLETE")
    print(f" Mean Dice: {df['Dice'].mean():.4f}")
    print(f" Mean HD95: {df['Hausdorff95'].mean():.4f}")
    print(f" CSV Saved to: {csv_path}")
    print(f" Native 3D Reconstructions in: {nifti_dir}")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()