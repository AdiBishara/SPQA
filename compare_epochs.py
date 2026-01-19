import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OMP error

import torch
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import sys
import importlib.util
import cv2

# --- SETTINGS ---
PROJECT_ROOT = r"C:\Users\97252\SPQA"
# 1. We need BOTH the image and the mask
IMAGE_PATH = os.path.join(PROJECT_ROOT, "synthstrip_data_v1.5", "asl_epi_121", "image.nii.gz")
MASK_PATH = os.path.join(PROJECT_ROOT, "synthstrip_data_v1.5", "asl_epi_121", "mask.nii.gz")

EPOCH_A_PATH = os.path.join(PROJECT_ROOT, "logs", "gan_checkpoints", "dae_gan_epoch_75.pth")
EPOCH_B_PATH = os.path.join(PROJECT_ROOT, "logs", "ae_checkpoints", "dae_normal_epoch_75.pth")
DAE_FILE_PATH = os.path.join(PROJECT_ROOT, "utils", "models", "vae.py")

# --- MODEL CONFIG ---
MODEL_CONFIG = {
    'in_channels': 2,  # Expects [Image, Mask]
    'out_channels': 1,
    'image_size': [512, 512]  # Corrected size
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODEL CLASS ---
if not os.path.exists(DAE_FILE_PATH):
    print(f"CRITICAL ERROR: Could not find model file at: {DAE_FILE_PATH}")
    sys.exit(1)

try:
    spec = importlib.util.spec_from_file_location("dae_module", DAE_FILE_PATH)
    dae_module = importlib.util.module_from_spec(spec)
    sys.modules["dae_module"] = dae_module
    spec.loader.exec_module(dae_module)
    DAE = dae_module.DAE
except Exception as e:
    print(f"Error loading python file: {e}")
    sys.exit(1)


def load_nifti_slice(path, normalize=True):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None
    try:
        nifti = nib.load(path)
        data = nifti.get_fdata()
        if len(data.shape) == 4: data = data[:, :, :, 0]

        # Get Middle Slice
        mid_slice = data.shape[2] // 2
        slice_data = data[:, :, mid_slice]

        if normalize:
            # Normalize to 0-1
            slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data) + 1e-8)

        # Resize to 512x512
        slice_resized = cv2.resize(slice_data, (512, 512), interpolation=cv2.INTER_NEAREST)
        return slice_resized
    except Exception as e:
        print(f"Error reading NIfTI {path}: {e}")
        return None


def load_and_infer(checkpoint_path, input_tensor):
    if not os.path.exists(checkpoint_path):
        return np.zeros((512, 512))

    model = DAE(**MODEL_CONFIG).to(DEVICE)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'g_state_dict' in checkpoint:  # Handle GAN format
            model.load_state_dict(checkpoint['g_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error weights: {e}")
        return np.zeros((512, 512))

    model.eval()
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor))
    return output.cpu().numpy().squeeze()


def compare_results():
    print(f"--- Visualizing With Mask Input ---")

    # 1. Load Image and Mask
    print("Loading Image...")
    img_slice = load_nifti_slice(IMAGE_PATH, normalize=True)
    print("Loading Mask...")
    mask_slice = load_nifti_slice(MASK_PATH, normalize=True)  # Masks are usually 0 or 1 already

    if img_slice is None or mask_slice is None:
        print("Could not load data. Check paths.")
        return

    # 2. Corrupt the mask slightly (Simulate Training Conditions)
    # We flip 10% of pixels to see if the model fixes it
    noisy_mask = mask_slice.copy()
    noise = np.random.random(noisy_mask.shape)
    mask_indices = noise < 0.1  # 10% noise
    noisy_mask[mask_indices] = 1 - noisy_mask[mask_indices]

    # 3. Create Input Tensor: [1, 2, 512, 512]
    # Channel 0: Image
    # Channel 1: Noisy Mask
    t_img = torch.from_numpy(img_slice).float().unsqueeze(0).unsqueeze(0)
    t_mask = torch.from_numpy(noisy_mask).float().unsqueeze(0).unsqueeze(0)

    input_tensor = torch.cat([t_img, t_mask], dim=1).to(DEVICE)

    # 4. Inference
    print("Running GAN...")
    out_gan = load_and_infer(EPOCH_A_PATH, input_tensor)

    print("Running DAE...")
    out_normal = load_and_infer(EPOCH_B_PATH, input_tensor)

    # 5. Plot
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.title("Input: Image")
    plt.imshow(img_slice, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Input: Corrupted Mask")
    plt.imshow(noisy_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("GAN Output 50")
    plt.imshow(out_gan, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("DAE Output 50")
    plt.imshow(out_normal, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("Done.")


if __name__ == "__main__":
    compare_results()