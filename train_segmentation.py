import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.seeding import fix_seeds
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.models.unet_dropout import UNet
from losses.losses import dice_loss, bce_criterion

def train_segmentor():
    # 1. Load Configuration and Environment
    config_path = r"C:\Users\97252\SPQA\params\config.yaml"
    config = load_config(config_path) [cite: 10]
    fix_seeds(config.get('seed', 42)) [cite: 13]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loader (Few-Shot NIfTI)
    # This engine handles the 3D-to-2D slicing and heavy augmentation
    train_dataset = NiftiFewShotDataset(
        data_root=config['Data']['raw_data_root'],
        id_file=config['Data']['few_shot_ids'],
        is_train=True,
        image_size=config['model']['image_size'][0]
    )
    train_loader = DataLoader(train_dataset, batch_size=config['Train']['batch_size'], shuffle=True)

    # 3. Model Initialization
    # We use the UNet with dropout for later MC uncertainty estimation
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['n_classes'],
        channels=config['model']['channels']
    ).to(device) [cite: 14]

    optimizer = optim.Adam(model.parameters(), lr=config['Train']['learning_rate'])

    # 4. Training Loop
    print("--- Starting UNet Segmentor Training ---")
    model.train()
    for epoch in range(config['Train']['epochs']):
        epoch_loss = 0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Combine BCE and Dice Loss for robust segmentation 
            loss_bce = bce_criterion(outputs, masks)
            loss_dice = dice_loss(torch.sigmoid(outputs), masks)
            total_loss = loss_bce + loss_dice
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        print(f"Epoch {epoch+1}/{config['Train']['epochs']} - Avg Loss: {epoch_loss/len(train_loader):.4f}")

    # 5. Save the Segmentor Checkpoint
    save_dir = config['Train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_unet_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Segmentor training complete. Model saved at: {save_path}")

if __name__ == "__main__":
    train_segmentor()