import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.models.unet_dropout import UNet
from losses.losses import dice_loss


def train_unet():
    # 1. Load Configuration
    config_path = r"C:\Users\97252\SPQA\params\config.yaml"
    config = load_config(config_path)
    print("DEBUG: Loaded Strides:", config['model']['strides'])
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.benchmark = True
        print(f"✅ Training on GPU: {torch.cuda.get_device_name(0)} (High Res Mode)")
    else:
        device = torch.device("cpu")

    # Save Path
    save_dir = r"C:\Users\97252\SPQA\logs\checkpoints"
    os.makedirs(save_dir, exist_ok=True)


    # 2. Data Loader
    dataset = NiftiFewShotDataset(
        data_root=config['Data']['raw_data_root'],
        id_file=config['Data']['training_ids'],
        image_size=config['model']['image_size'][0]  # This passes 512
    )

    loader = DataLoader(
        dataset,
        batch_size=config['Train']['batch_size'],  # If you get "Out of Memory", lower this to 4 or 2
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 3. Model Initialization
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['n_classes'],
        channels=config['model']['channels'],
        strides=config['model']['strides'],
        dropout=config['model']['dropout_rate']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['Train']['learning_rate_gen'])
    criterion = dice_loss

    # 4. Training Loop
    model.train()

    for epoch in range(config['Train']['epochs']):
        epoch_loss = 0

        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = torch.sigmoid(model(images))
            loss = criterion(outputs, masks).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        # Save Checkpoint
        checkpoint_name = f"best_unet_model_epoch{epoch + 1}.pth"
        save_path = os.path.join(save_dir, checkpoint_name)
        torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch + 1}/{config['Train']['epochs']} | Loss: {avg_loss:.4f} -> Saved: {checkpoint_name}")

    print(f"Training Complete.")


if __name__ == "__main__":
    train_unet()