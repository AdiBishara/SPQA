import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.models.dae import DAE, Discriminator
from losses.losses import dice_loss

def train_adversarial_ae():
    # 1. Setup Environment
    config_path = r"C:\Users\97252\SPQA\params\config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loader
    dataset = NiftiFewShotDataset(
        data_root=config['Data']['raw_data_root'], 
        id_file=config['Data']['few_shot_ids'],
        image_size=config['model']['image_size'][0]
    )
    loader = DataLoader(dataset, batch_size=config['Train']['batch_size'], shuffle=True)

    # 3. Model Initialization
    # Generator (DAE): Takes [Image, NoisyMask] -> Outputs [CleanMask]
    generator = DAE(
        in_channels=2,  # <--- CHANGED: Matches train_QC_AE logic
        out_channels=1, 
        image_size=config['model']['image_size']
    ).to(device)
    
    # Discriminator: Takes [Mask] -> Outputs [Real/Fake Score]
    # We feed it 1 channel (the mask) to check if it looks like a real brain shape
    discriminator = Discriminator(in_channels=1).to(device)

    # 4. Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=config['Train']['learning_rate'])
    opt_D = optim.Adam(discriminator.parameters(), lr=config['Train']['learning_rate'])
    
    # Adversarial Loss (BCE)
    adversarial_criterion = nn.BCEWithLogitsLoss() 

    print("--- Starting Adversarial QC Training (DAE + GAN) ---")

    for epoch in range(config['Train']['epochs']):
        running_loss_g = 0.0
        running_loss_d = 0.0
        
        for images, clean_masks, _ in loader:
            images, clean_masks = images.to(device), clean_masks.to(device)
            batch_size = images.size(0)

            # --- STEP A: PREPARE CORRUPTED INPUT (The "Operations") ---
            corrupted_target = clean_masks.clone()
            noise = torch.rand_like(clean_masks)
            mask_noise = noise < 0.1 # Flip 10% of pixels
            corrupted_target[mask_noise] = 1 - corrupted_target[mask_noise]

            # Input to DAE: MRI + Bad Mask
            dae_input = torch.cat([images, corrupted_target], dim=1)

            # --- STEP B: TRAIN DISCRIMINATOR ---
            opt_D.zero_grad()
            
            # 1. Train on Real Data (The Clean Masks from Dataset)
            real_labels = torch.ones(batch_size, 1).to(device)
            # Add noise to labels for stability (e.g. 0.9 instead of 1.0)
            real_labels = real_labels * 0.9 
            
            output_real = discriminator(clean_masks)
            loss_d_real = adversarial_criterion(output_real, real_labels)

            # 2. Train on Fake Data (The Output of DAE)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Generate fake mask (detach so we don't train G yet)
            fake_masks = torch.sigmoid(generator(dae_input))
            output_fake = discriminator(fake_masks.detach())
            loss_d_fake = adversarial_criterion(output_fake, fake_labels)
            
            loss_D = (loss_d_real + loss_d_fake) / 2
            loss_D.backward()
            opt_D.step()

            # --- STEP C: TRAIN GENERATOR (DAE) ---
            opt_G.zero_grad()
            
            # We want the Discriminator to think these fake masks are REAL (label=1)
            output_fake_for_g = discriminator(fake_masks)
            
            # 1. Adversarial Loss (Fool the discriminator)
            loss_g_adv = adversarial_criterion(output_fake_for_g, torch.ones_like(real_labels))
            
            # 2. Reconstruction Loss (Actually look like the target mask)
            loss_g_recon = dice_loss(fake_masks, clean_masks).mean()
            
            # Combine losses (0.01 weight for adv is standard to prevent mode collapse)
            loss_G = loss_g_recon + (0.01 * loss_g_adv)
            
            loss_G.backward()
            opt_G.step()

            running_loss_g += loss_G.item()
            running_loss_d += loss_D.item()

        print(f"Epoch {epoch+1} | G Loss: {running_loss_g/len(loader):.4f} | D Loss: {running_loss_d/len(loader):.4f}")

    # 5. Save Checkpoint
    save_path = config['DAE']['checkpoint_path'].replace(".pth", "_gan.pth")
    torch.save(generator.state_dict(), save_path)
    print(f"GAN-DAE Model saved to: {save_path}")

if __name__ == "__main__":
    train_adversarial_ae()