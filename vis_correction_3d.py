import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from utils.config import load_config
from utils.data.nifti_loader import NiftiDataset
from utils.models.vae import VAE3D
import glob
import re


def get_latest_checkpoint(save_dir):
    checkpoints = glob.glob(os.path.join(save_dir, "vae_epoch_*.pth"))
    if not checkpoints: return None
    latest_ckpt = max(checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    return latest_ckpt


def morphological_corruption(mask, kernel_size=3, iters=3, mode='erode'):
    """
    Deterministically corrupts a mask for visualization.
    """
    pad = kernel_size // 2
    corrupted = mask.clone()

    if mode == "dilate":
        for _ in range(iters):
            corrupted = F.max_pool3d(corrupted, kernel_size=kernel_size, stride=1, padding=pad)
    else:  # erode
        corrupted = -corrupted
        for _ in range(iters):
            corrupted = F.max_pool3d(corrupted, kernel_size=kernel_size, stride=1, padding=pad)
        corrupted = -corrupted
    return corrupted


def save_nifti(data_tensor, reference_path, output_path):
    # Load original to get the affine matrix (position in 3D space)
    ref_img = nib.load(reference_path)
    affine = ref_img.affine

    # Convert Tensor -> Numpy
    if torch.is_tensor(data_tensor):
        data = data_tensor.detach().cpu().numpy()
    else:
        data = data_tensor

    # Remove Batch and Channel dimensions [1, 1, D, H, W] -> [D, H, W]
    data = np.squeeze(data)

    # Save
    nifti_img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(nifti_img, output_path)
    print(f"Saved: {os.path.basename(output_path)}")


def visualize_correction_3d():
    # 1. SETUP
    config_path = r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\results_plots"
    os.makedirs(save_dir, exist_ok=True)

    # 2. LOAD MODEL
    print("Loading VAE...")
    vae = VAE3D(
        in_channels=2,
        out_channels=1,
        image_size=config['model']['image_size'],
        latent_dim=config['model']['latent_dim']
    ).to(device)

    # Load Latest Checkpoint
    ckpt_dir = r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs\vae_checkpoints"
    ckpt_path = get_latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        print("No checkpoints found!")
        return

    print(f"Loading Weights: {os.path.basename(ckpt_path)}")
    vae.load_state_dict(torch.load(ckpt_path, map_location=device))
    vae.eval()

    # 3. GET ONE SAMPLE
    dataset = NiftiDataset(
        img_dir=config['Data']['raw_data_root'],
        list_path=config['Data']['training_ids'],
        image_size=config['model']['image_size'],
        is_train=False
    )

    # Grab the first subject in the list
    idx = 0
    sample = dataset[idx]

    # Get original file path for Affine/Header info
    original_mask_path = dataset.file_list[idx][1]  # (img_path, mask_path, id)

    image = sample['image'].to(device).unsqueeze(0)  # [1, 1, D, H, W]
    clean_mask = sample['mask'].to(device).unsqueeze(0)

    # 4. RUN EROSION TEST
    print("\n--- TEST 1: Healing Shrinkage (Erosion) ---")
    eroded_mask = morphological_corruption(clean_mask, iters=5, mode='erode')

    # Infer
    input_eroded = torch.cat([image, eroded_mask], dim=1)
    with torch.no_grad():
        recon_erode, _, _ = vae(input_eroded)
        recon_erode = torch.sigmoid(recon_erode)
        recon_erode = (recon_erode > 0.5).float()  # Threshold to binary

    # 5. RUN DILATION TEST
    print("--- TEST 2: Trimming Growth (Dilation) ---")
    dilated_mask = morphological_corruption(clean_mask, iters=5, mode='dilate')

    # Infer
    input_dilated = torch.cat([image, dilated_mask], dim=1)
    with torch.no_grad():
        recon_dilate, _, _ = vae(input_dilated)
        recon_dilate = torch.sigmoid(recon_dilate)
        recon_dilate = (recon_dilate > 0.5).float()

    # 6. SAVE EVERYTHING AS NIFTI
    print("\n--- Saving 3D Volumes ---")

    save_nifti(clean_mask, original_mask_path, os.path.join(save_dir, "01_original_truth.nii.gz"))

    # Erosion Set
    save_nifti(eroded_mask, original_mask_path, os.path.join(save_dir, "02_input_eroded.nii.gz"))
    save_nifti(recon_erode, original_mask_path, os.path.join(save_dir, "03_output_healed.nii.gz"))

    # Dilation Set
    save_nifti(dilated_mask, original_mask_path, os.path.join(save_dir, "04_input_dilated.nii.gz"))
    save_nifti(recon_dilate, original_mask_path, os.path.join(save_dir, "05_output_trimmed.nii.gz"))

    print("\nDone. Drag these files into ITK-SNAP to view.")


if __name__ == "__main__":
    visualize_correction_3d()