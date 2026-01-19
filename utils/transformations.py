import torch
from torchvision import transforms

# Forward transformation for preprocessing before inference
forward_transform = transforms.Compose([
    transforms.Resize((512, 512), antialias=True),  # Changed from 512 to match Config
])

# Inverse transformation to map predictions back to original shape (if needed)
inverse_transform = transforms.Compose([
    transforms.Resize((512, 512), antialias=True),  # Matches Brain MRI resolution
])