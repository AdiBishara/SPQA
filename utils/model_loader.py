import os
import torch
import sys

# Try to import natsort, fallback to standard sort if missing
try:
    import natsort


    def sorted_files(file_list):
        return natsort.natsorted(file_list)
except ImportError:
    print("Warning: 'natsort' library not found. Using standard sort.")


    def sorted_files(file_list):
        return sorted(file_list)

# Import the custom UNet
from utils.models.unet_dropout import UNet


def find_model_file(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    all_files = os.listdir(directory)
    model_files = [f for f in all_files if f.endswith(('.pth', '.pt'))]
    if not model_files:
        raise FileNotFoundError(f"No .pth or .pt model files found in {directory}")
    return os.path.join(directory, model_files[0])


def load_unet_model(model_details):
    """
    Load a single UNet model with FORCED Stride 1 settings.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n--- LOADING UNET ---")

    # 2. INITIALIZE MODEL
    net = UNet(
        spatial_dims=2,
        in_channels=model_details['in_channels'],
        out_channels=model_details['n_classes'],
        channels=model_details['channels'],
        dropout=model_details['dropout_rate'],
        strides=model_details['strides']
    )

    # 3. FIND CHECKPOINT
    restore_path = model_details['model_path']
    if not os.path.isfile(restore_path):
        try:
            restore_path = find_model_file(restore_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No model found in {restore_path}")

    print(f"Loading weights from: {restore_path}")

    # 4. LOAD WEIGHTS
    net.load_state_dict(torch.load(restore_path, map_location=device), strict=True)

    net.to(device)
    net.eval()
    print("--- MODEL LOADED SUCCESSFULLY ---\n")

    return net, device