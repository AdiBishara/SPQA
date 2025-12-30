import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.seeding import fix_seeds
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.models.dae import DAE
from losses.losses import dice_loss

def train_standard_ae():
    # 1. Setup Environment
    config = load_config(r"C:\Users\97252\SPQA\params\config.yaml")
    fix_seeds(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Few-Shot Data Loading
    dataset = NiftiFewShotDataset(
        data_root=config['Data']['raw_data_root'],
        id_file=config['Data']['training_ids'],
        is_train=True,
        image_size=config['model']['image_size'][0]
    )
    loader = DataLoader(dataset, batch_size=config['Train']['batch_size'], shuffle=True)

    # 3. Model Initialization
    model = DAE(
        in_channels=2, 
        out_channels=1, 
        image_size=config['model']['image_size']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['Train']['learning_rate'])

    # 4. Standard Training Loop
    model.train()
    print("Starting Standard DAE Training...")
    for epoch in range(config['Train']['epochs']):
        epoch_loss = 0
        for corrupted, target, _ in loader:
            corrupted, target = corrupted.to(device), target.to(device)
            corrupted_target = target.clone()
            noise = torch.rand_like(target)
            mask_noise = noise < 0.1 
            corrupted_target[mask_noise] = 1 - corrupted_target[mask_noise]
            dae_input = torch.cat([images, corrupted_target], dim=1)
            optimizer.zero_grad()
            output = torch.sigmoid(model(dae_input))
            # Focus on Dice loss for shape overlap
            loss = dice_loss(output, target).mean() 
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{config['Train']['epochs']} - Loss: {epoch_loss/len(loader):.4f}")

    # 5. Save Checkpoint
    save_path = config['DAE']['checkpoint_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully at {save_path}")

if __name__ == "__main__":
    train_standard_ae()