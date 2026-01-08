import os
import torch
import numpy as np
import nibabel as nib
import cv2
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class NiftiFewShotDataset(Dataset):
    def __init__(self, data_root, id_file, is_train=True, image_size=512, transform=None):
        """
        Custom Dataset for Few-Shot learning from NIfTI volumes.
        """
        self.data_root = data_root
        self.is_train = is_train
        self.image_size = image_size
        self.transform = transform  # <--- FIXED: Now stores the transform

        with open(id_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines() if line.strip()]

        self.volumes = []
        self.slice_map = []

        print(f"--- Few-Shot Loader Initialized ---")
        print(f"Loading {len(self.ids)} volumes for {'Training' if is_train else 'Testing'}")

        for vol_idx, subj_id in enumerate(self.ids):
            img_path = os.path.join(data_root, subj_id, "image.nii.gz")
            mask_path = os.path.join(data_root, subj_id, "mask.nii.gz")

            if not os.path.exists(img_path):
                img_path = img_path.replace(".nii.gz", ".nii")
                mask_path = mask_path.replace(".nii.gz", ".nii")

            img_vol = nib.load(img_path).get_fdata()
            mask_vol = nib.load(mask_path).get_fdata()

            # Normalize
            img_vol = (img_vol - img_vol.min()) / (img_vol.max() - img_vol.min() + 1e-8)
            mask_vol = (mask_vol > 0).astype(np.float32)

            self.volumes.append((img_vol, mask_vol))

            # Map every slice for sequential access
            num_slices = img_vol.shape[2]
            for s in range(num_slices):
                self.slice_map.append((vol_idx, s))

    def __len__(self):
        return 1000 if self.is_train else len(self.slice_map)

    def apply_augmentation(self, image, mask):
        img_pil = TF.to_pil_image(torch.from_numpy(image).unsqueeze(0))
        mask_pil = TF.to_pil_image(torch.from_numpy(mask).unsqueeze(0))

        if random.random() > 0.5:
            angle = random.randint(-25, 25)
            img_pil = TF.rotate(img_pil, angle)
            mask_pil = TF.rotate(mask_pil, angle)

        if random.random() > 0.5:
            img_pil = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)

        return TF.to_tensor(img_pil), TF.to_tensor(mask_pil)

    def __getitem__(self, idx):
        if self.is_train:
            # Random selection for training
            vol_idx = random.randint(0, len(self.volumes) - 1)
            img_vol, mask_vol = self.volumes[vol_idx]
            num_slices = img_vol.shape[2]
            # Random slice avoiding top/bottom
            slice_idx = random.randint(int(num_slices * 0.2), int(num_slices * 0.8))
        else:
            # Sequential selection for testing/recon
            vol_idx, slice_idx = self.slice_map[idx]
            img_vol, mask_vol = self.volumes[vol_idx]

        img_slice = img_vol[:, :, slice_idx]
        mask_slice = mask_vol[:, :, slice_idx]

        # Base Resize (using OpenCV)
        img_slice = cv2.resize(img_slice, (self.image_size, self.image_size))
        mask_slice = cv2.resize(mask_slice, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            img_tensor, mask_tensor = self.apply_augmentation(img_slice, mask_slice)
        else:
            img_tensor = torch.from_numpy(img_slice).unsqueeze(0).float()
            mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).float()

        # FIXED: Apply External Transform (e.g., Resize from run_QC.py)
        if self.transform:
            img_tensor = self.transform(img_tensor)

            # CRITICAL CHECK: If the transform changed the image size (e.g. 512 -> 256),
            # we MUST resize the mask to match, otherwise evaluation metrics will crash.
            if mask_tensor is not None and (img_tensor.shape[-1] != mask_tensor.shape[-1]):
                mask_tensor = TF.resize(
                    mask_tensor,
                    img_tensor.shape[-2:],
                    interpolation=TF.InterpolationMode.NEAREST
                )

        # Returns subject ID in name so evaluation.py knows when a volume ends
        return img_tensor, mask_tensor, f"{self.ids[vol_idx]}_slice_{slice_idx}"