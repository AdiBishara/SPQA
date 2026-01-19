import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset


class NiftiDataset(Dataset):
    def __init__(self, img_dir, list_path, is_train=True, image_size=(256, 256, 256), transform=None):
        self.data_root = img_dir
        self.is_train = is_train

        if isinstance(image_size, int):
            self.target_size = (image_size, image_size, image_size)
        elif isinstance(image_size, list):
            self.target_size = tuple(image_size)
        else:
            self.target_size = image_size

        with open(list_path, 'r') as f:
            self.ids = [line.strip() for line in f.readlines() if line.strip()]

        self.file_list = []
        for subject_id in self.ids:
            img_path = os.path.join(self.data_root, subject_id, "image.nii.gz")
            mask_path = os.path.join(self.data_root, subject_id, "mask.nii.gz")

            if not os.path.exists(img_path):
                img_path = img_path.replace(".nii.gz", ".nii")
                mask_path = mask_path.replace(".nii.gz", ".nii")

            self.file_list.append((img_path, mask_path, subject_id))

        print(f"--- Fast 3D Loader Initialized ({'Train' if is_train else 'Test'}) ---")

    def __len__(self):
        return len(self.ids)

    def _resize_volume(self, img, mask):
        d, h, w = img.shape
        td, th, tw = self.target_size

        # Padding
        pad_d, pad_h, pad_w = max(0, td - d), max(0, th - h), max(0, tw - w)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pd1, pd2 = pad_d // 2, pad_d - (pad_d // 2)
            ph1, ph2 = pad_h // 2, pad_h - (pad_h // 2)
            pw1, pw2 = pad_w // 2, pad_w - (pad_w // 2)

            img = np.pad(img, ((pd1, pd2), (ph1, ph2), (pw1, pw2)), mode='constant')
            mask = np.pad(mask, ((pd1, pd2), (ph1, ph2), (pw1, pw2)), mode='constant')
            d, h, w = img.shape

        # Center Crop
        if d > td or h > th or w > tw:
            z, y, x = (d - td) // 2, (h - th) // 2, (w - tw) // 2
            img = img[z:z + td, y:y + th, x:x + tw]
            mask = mask[z:z + td, y:y + th, x:x + tw]

        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path, subject_id = self.file_list[idx]

        try:
            img = nib.load(img_path).get_fdata().astype(np.float32)
            mask = nib.load(mask_path).get_fdata().astype(np.float32)

            # --- CRITICAL FIX: FORCE BINARY MASK ---
            # If mask has values like 255, this sets them to 1.0
            # If mask is 0.001 (soft), this sets it to 1.0
            mask = (mask > 0).astype(np.float32)
            # ---------------------------------------

            # Normalize Image
            if np.max(img) > 0:
                img = (img - np.min(img)) / (np.max(img) - np.min(img))

            img, mask = self._resize_volume(img, mask)

            img_tensor = torch.from_numpy(img).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)

            return {'image': img_tensor, 'mask': mask_tensor, 'id': subject_id}

        except Exception as e:
            print(f"Error loading {subject_id}: {e}")
            return self.__getitem__((idx + 1) % len(self.ids))