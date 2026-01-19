import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.models.vae import DAE, Discriminator
from losses.losses import dice_loss
import os
import numpy as np


def train_adversarial_ae():
    # 1. Load Configuration
    config_path = r"C:\Users\97252\SPQA\params\config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = r"C:\Users\97252\SPQA\logs\gan_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print(f"--- Starting Adversarial Training ---")
    print(f"LOGGING TO: {save_dir}")

    # 2. Data Loader
    dataset = NiftiFewShotDataset(
        data_root=config['Data']['raw_data_root'],
        id_file=config['Data']['training_ids'],
        image_size=config['model']['image_size'][0]
    )
    loader = DataLoader(dataset, batch_size=config['Train']['batch_size'], shuffle=True)

    # 3. Model Initialization
    generator = DAE(
        in_channels=2,
        out_channels=1,
        image_size=config['model']['image_size']
    ).to(device)

    discriminator = Discriminator(
        in_channels=1,
    ).to(device)

    adversarial_criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(generator.parameters(), lr=config['Train']['learning_rate_gen'])
    opt_D = optim.Adam(discriminator.parameters(), lr=config['Train']['learning_rate_disc'])

    print(
        f"--> Optimizer Setup | Gen LR: {config['Train']['learning_rate_gen']} | Disc LR: {config['Train']['learning_rate_disc']}")

    # 4. Training Loop
    for epoch in range(config['Train']['epochs']):
        running_loss_g = 0.0
        running_loss_d = 0.0

        for i, batch in enumerate(loader):
            images, clean_masks, _ = batch
            images, clean_masks = images.to(device), clean_masks.to(device)
            batch_size = images.size(0)

            # --- NEW: Block Corruption Logic (Cutout) ---
            corrupted_masks = clean_masks.clone()

            # Apply to each item in batch
            for b in range(corrupted_masks.size(0)):
                h_box = np.random.randint(30, 80)  # Size of the cut
                w_box = np.random.randint(30, 80)

                # Random location
                y_loc = np.random.randint(0, images.shape[-2] - h_box)
                x_loc = np.random.randint(0, images.shape[-1] - w_box)

                # 50% Erase (0), 50% Add Fake Block (1)
                val = 0.0 if np.random.rand() > 0.5 else 1.0
                corrupted_masks[b, 0, y_loc:y_loc + h_box, x_loc:x_loc + w_box] = val
            # ----------------------------------------------

            dae_input = torch.cat([images, corrupted_masks], dim=1)

            # A. Train Discriminator
            opt_D.zero_grad()

            # Real
            real_labels = torch.ones(batch_size, 1).to(device)
            output_real = discriminator(clean_masks)
            loss_d_real = adversarial_criterion(output_real, real_labels)

            # Fake
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_masks = torch.sigmoid(generator(dae_input))
            output_fake = discriminator(fake_masks.detach())
            loss_d_fake = adversarial_criterion(output_fake, fake_labels)

            loss_D = (loss_d_real + loss_d_fake) / 2
            loss_D.backward()
            opt_D.step()

            # B. Train Generator
            opt_G.zero_grad()
            output_fake_for_g = discriminator(fake_masks)

            # Generator wants discriminator to say "1" (Real)
            loss_g_adv = adversarial_criterion(output_fake_for_g, torch.ones_like(real_labels))
            loss_g_recon = dice_loss(fake_masks, clean_masks).mean()

            # Weighted Loss: 1.0 Recon + 0.01 Adversarial
            loss_G = loss_g_recon + (0.01 * loss_g_adv)
            loss_G.backward()
            opt_G.step()

            running_loss_g += loss_G.item()
            running_loss_d += loss_D.item()

        avg_g_loss = running_loss_g / len(loader)
        avg_d_loss = running_loss_d / len(loader)

        print(f"Epoch {epoch + 1} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")

        # Save Checkpoint
        filename = f"dae_gan_epoch_{epoch + 1}.pth"
        save_path = os.path.join(save_dir, filename)
        torch.save(generator.state_dict(), save_path)
        print(f"   -> Saved: {filename}")


if __name__ == "__main__":
    train_adversarial_ae()