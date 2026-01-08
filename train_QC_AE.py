import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.seeding import fix_seeds
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.models.dae import DAE
from losses.losses import dice_loss
import datetime
import numpy as np


def train_standard_ae():
    # 1. Setup Environment
    config = load_config(r"C:\Users\97252\SPQA\params\config.yaml")
    fix_seeds(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = r"C:\Users\97252\SPQA"
    save_dir = os.path.join(project_root, "logs", "ae_checkpoints")

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    print(f"--- Starting Standard DAE Training ---")
    print(f"LOGGING TO: {save_dir}")

    # 2. Few-Shot Data Loading
    dataset = NiftiFewShotDataset(
        data_root=config['Data']['raw_data_root'],
        id_file=config['Data']['training_ids'],
        image_size=config['model']['image_size'][0]
    )
    loader = DataLoader(dataset, batch_size=config['Train']['batch_size'], shuffle=True)

    # 3. Model Initialization
    model = DAE(
        in_channels=2,
        out_channels=1,
        image_size=config['model']['image_size']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['Train']['learning_rate_gen'])

    # 4. Standard Training Loop
    model.train()

    for epoch in range(config['Train']['epochs']):
        epoch_loss = 0

        for images, target, _ in loader:
            images, target = images.to(device), target.to(device)

            corrupted_target = target.clone()

            for i in range(corrupted_target.size(0)):
                h_box = np.random.randint(30, 80)
                w_box = np.random.randint(30, 80)

                y_loc = np.random.randint(0, images.shape[-2] - h_box)
                x_loc = np.random.randint(0, images.shape[-1] - w_box)

                val = 0.0 if np.random.rand() > 0.5 else 1.0
                corrupted_target[i, 0, y_loc:y_loc + h_box, x_loc:x_loc + w_box] = val

            # Prepare DAE Input: Image + Bad Mask
            dae_input = torch.cat([images, corrupted_target], dim=1)

            optimizer.zero_grad()
            output = torch.sigmoid(model(dae_input))

            # Focus on Dice loss for shape overlap
            loss = dice_loss(output, target).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        # Save Checkpoint
        filename = f"dae_normal_epoch_{epoch + 1}.pth"
        save_path = os.path.join(save_dir, filename)
        torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch + 1}/{config['Train']['epochs']} - Loss: {avg_loss:.4f} -> Saved: {filename}")


if __name__ == "__main__":
    train_standard_ae()