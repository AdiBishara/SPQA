import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.models.dae import DAE, Discriminator
from losses.losses import dice_loss
import os
import datetime


def train_adversarial_ae():
    # 1. Load Configuration
    config_path = r"C:\Users\97252\SPQA\params\config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_root = r"C:\Users\97252\SPQA"
    base_save_dir = os.path.join(project_root, "GAN_AE_checkpoints")
    run_folder = os.path.join(base_save_dir, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    print(f"--- Starting Adversarial Training ---")
    print(f"LOGGING TO: {run_folder}")

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

    discriminator = Discriminator(in_channels=1).to(device)

    # 4. Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=config['Train']['learning_rate'])
    opt_D = optim.Adam(discriminator.parameters(), lr=config['Train']['learning_rate'])

    adversarial_criterion = nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    for epoch in range(config['Train']['epochs']):
        running_loss_g = 0.0
        running_loss_d = 0.0

        for images, clean_masks, _ in loader:
            images, clean_masks = images.to(device), clean_masks.to(device)
            batch_size = images.size(0)

            # A. Prepare Input
            corrupted_target = clean_masks.clone()
            noise = torch.rand_like(clean_masks)
            mask_noise = noise < 0.1
            corrupted_target[mask_noise] = 1 - corrupted_target[mask_noise]
            dae_input = torch.cat([images, corrupted_target], dim=1)

            # B. Train Discriminator
            opt_D.zero_grad()

            real_labels = torch.ones(batch_size, 1).to(device) * 0.9
            output_real = discriminator(clean_masks)
            loss_d_real = adversarial_criterion(output_real, real_labels)

            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_masks = torch.sigmoid(generator(dae_input))
            output_fake = discriminator(fake_masks.detach())
            loss_d_fake = adversarial_criterion(output_fake, fake_labels)

            loss_D = (loss_d_real + loss_d_fake) / 2
            loss_D.backward()
            opt_D.step()

            # C. Train Generator
            opt_G.zero_grad()
            output_fake_for_g = discriminator(fake_masks)
            loss_g_adv = adversarial_criterion(output_fake_for_g, torch.ones_like(real_labels))
            loss_g_recon = dice_loss(fake_masks, clean_masks).mean()

            loss_G = loss_g_recon + (0.01 * loss_g_adv)
            loss_G.backward()
            opt_G.step()

            running_loss_g += loss_G.item()
            running_loss_d += loss_D.item()

        avg_g_loss = running_loss_g / len(loader)
        avg_d_loss = running_loss_d / len(loader)

        print(f"Epoch {epoch + 1} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")

        # --- SAVE CHECKPOINT EVERY EPOCH ---
        filename = f"dae_gan_epoch_{epoch + 1}.pth"
        save_path = os.path.join(run_folder, filename)

        torch.save(generator.state_dict(), save_path)
        print(f"   -> Saved: {filename}")

    print(f"Training Complete. All checkpoints saved in: {run_folder}")


if __name__ == "__main__":
    train_adversarial_ae()