import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.models.dae import DAE, Discriminator
# Import the separated GAN logic
from utils.trainers.gan_trainer import train_one_epoch 

def train_adversarial_ae():
    config = load_config(r"C:\Users\97252\SPQA\params\config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dataset = NiftiFewShotDataset(data_root=config['Data']['raw_data_root'], id_file=config['Data']['few_shot_ids'])
    loader = DataLoader(dataset, batch_size=config['Train']['batch_size'], shuffle=True)

    # Models: Generator (DAE) and Discriminator
    model = DAE(image_size=config['model']['image_size']).to(device)
    discriminator = Discriminator(in_channels=1).to(device)

    opt_G = optim.Adam(model.parameters(), lr=config['Train']['learning_rate'])
    opt_D = optim.Adam(discriminator.parameters(), lr=config['Train']['learning_rate'])

    # Adversarial Training Loop using your trainer logic
    for epoch in range(config['Train']['epochs']):
        loss = train_one_epoch(model, discriminator, loader, opt_G, opt_D, device)
        print(f"GAN Epoch {epoch+1} - Generator Loss: {loss:.4f}")

    torch.save(model.state_dict(), config['DAE']['checkpoint_path'].replace(".pth", "_gan.pth"))

if __name__ == "__main__":
    train_adversarial_ae()