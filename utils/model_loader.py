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

# --- CONFIGURATION: Check your model file name here ---
try:
    # REVERT THIS: Use the standard UNet for segmentation
    from utils.models.unet_dropout import UNet
except ImportError:
    print("Critical Warning: Could not import UNet from utils.models.unet_dropout.")
    from .models.unet_dropout import UNet


def find_model_file(directory):
    """
    Helper to find the first valid .pth or .pt file in a directory.
    Ignores system files like .DS_Store.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_files = os.listdir(directory)
    # Filter for model files only
    model_files = [f for f in all_files if f.endswith(('.pth', '.pt'))]

    if not model_files:
        raise FileNotFoundError(f"No .pth or .pt model files found in {directory}")

    return os.path.join(directory, model_files[0])


def load_unet_model(model_details):
    """
    Load a single UNet model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the architecture
    net = UNet(
        in_channels=model_details['in_channels'],
        out_channels=model_details['n_classes'],
        channels=model_details['channels'],
        dropout=model_details['dropout_rate'],  # Ensure this matches config key
        strides = model_details['strides']
    )

    root_path = model_details['model_path']

    # FIX: Logic to handle direct file path or directory search
    if os.path.isfile(root_path):
        restore_path = root_path
    else:
        # If it's a folder, find the .pth file inside it
        try:
            restore_path = find_model_file(root_path)
        except FileNotFoundError:
            # Fallback: Try subfolders (old logic) if direct file not found
            subfolders = sorted_files([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
            if subfolders:
                restore_path = find_model_file(os.path.join(root_path, subfolders[-1]))
            else:
                raise FileNotFoundError(f"No model found in {root_path}")

    print(f"Loading model from: {restore_path}")
    net.load_state_dict(torch.load(restore_path, map_location=device))
    net.to(device)
    net.eval()

    return net, device

def load_unet_model_deep(model_details):
    """
    Load multiple UNet models (ensemble) from all subdirectories.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []

    root_path = model_details['model_path']
    # Get all subfolders (runs)
    run_folders = sorted_files([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])

    print(f"Loading ensemble of {len(run_folders)} models...")

    for subdir in run_folders:
        net = UNet(
            in_channels=model_details['in_channels'],
            n_classes=model_details['n_classes'],
            is_batchnorm=True,
            drop=0.3,
            channels=model_details['channels']
        )

        run_path = os.path.join(root_path, subdir)
        restore_path = find_model_file(run_path)

        net.load_state_dict(torch.load(restore_path, map_location=device))
        net.to(device)
        net.eval()  # Ensure eval mode for inference
        models.append(net)

    return models, device