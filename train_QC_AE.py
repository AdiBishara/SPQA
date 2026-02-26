import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
import re
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.seeding import fix_seeds
from utils.data.nifti_loader import NiftiDataset
from utils.models.vae import VAE3D
from losses.losses import VAELoss, dice_coefficient

def sync_weights_from_config(criterion, avg_dice, config):
    phases = config['phases']
    qualified = [(n, d) for n, d in phases.items() if avg_dice >= d['threshold']]
    best_name, best_data = max(qualified, key=lambda x: x[1]['threshold']) if qualified else (list(phases.keys())[0], list(phases.values())[0])
    criterion.w = {k: v for k, v in best_data.items() if k != 'threshold'}
    return best_name

def find_latest_checkpoint(save_dir):
    checkpoints = glob.glob(os.path.join(save_dir, "vae_epoch_*.pth"))
    if not checkpoints: return None, 0
    def extract_epoch(ckpt_path):
        m = re.search(r'vae_epoch_(\d+).pth', ckpt_path)
        return int(m.group(1)) if m else 0
    latest = max(checkpoints, key=extract_epoch)
    return latest, extract_epoch(latest)

class SequentialLogger(object):
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.terminal = sys.stdout
        num = len(glob.glob(os.path.join(log_dir, "vae_run_*.txt"))) + 1
        self.log = open(os.path.join(log_dir, f"vae_run_{num}.txt"), "a", encoding="utf-8")
        print(f"--- RUN 28 RECONSTRUCTION SESSION {num} ---")
    def write(self, message): self.terminal.write(message); self.log.write(message); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()

def save_visual_check(recon, target, image, epoch, save_dir, sid):
    try:
        os.makedirs(save_dir, exist_ok=True)
        d = image.shape[2]; slices = [int(d*0.25), int(d*0.5), int(d*0.75)]
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f"ID: {sid} | Epoch: {epoch}", fontsize=16)
        for i, s_idx in enumerate(slices):
            img_s = image[0,0,s_idx].detach().cpu().numpy()
            gt_s = target[0,0,s_idx].detach().cpu().numpy()
            pred_s = (torch.sigmoid(recon[0,0,s_idx]) > 0.5).detach().cpu().numpy()
            axes[i,0].imshow(img_s, cmap='gray'); axes[i,1].imshow(img_s, cmap='gray'); axes[i,1].imshow(gt_s, cmap='Greens', alpha=0.3)
            axes[i,2].imshow(img_s, cmap='gray'); axes[i,2].imshow(pred_s, cmap='Reds', alpha=0.3)
            for ax in axes[i]: ax.axis('off')
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}_{sid}.png")); plt.close(fig)
    except Exception as e: print(f"Visual Error: {e}")

def train_vae():
    config = load_config(r"C:\Users\Lab\OneDrive\Desktop\SPQA\params\config.yaml")
    fix_seeds(config['seed']); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir, save_dir, vis_dir = [os.path.join(r"C:\Users\Lab\OneDrive\Desktop\SPQA\logs", d) for d in ["training_logs", "vae_checkpoints", "visual_progress"]]
    for d in [log_dir, save_dir, vis_dir]: os.makedirs(d, exist_ok=True)
    sys.stdout = SequentialLogger(log_dir)
    model = VAE3D(in_channels=2, out_channels=1, latent_dim=config['model']['latent_dim']).to(device)
    ckpt, start_epoch = find_latest_checkpoint(save_dir)
    if ckpt: model.load_state_dict(torch.load(ckpt, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    criterion = VAELoss(config=config).to(device)
    bce_stable = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    scaler = GradScaler('cuda')
    loader = DataLoader(NiftiDataset(img_dir=config['data']['raw_data_root'], list_path=config['data']['training_ids'], image_size=config['model']['image_size']), batch_size=1, shuffle=True)

    rolling_dice = 0.0
    for epoch in range(start_epoch, config['train']['epochs']):
        model.train(); m = {'dice': 0, 'pxl': 0, 'count': 0}
        current_phase = sync_weights_from_config(criterion, rolling_dice, config)
        for i, batch in enumerate(loader):
            img, mask = batch['image'].to(device), (batch['mask'].to(device) > 0.5).float()
            sid = batch['id'][0]; corr = mask.clone() # Simple placeholder for discovery
            with autocast('cuda'):
                recon, mu, logvar = model(torch.cat([img, corr], dim=1))
                d_acc = dice_coefficient(torch.sigmoid(recon), mask)
                # Apply Run 28 Weights immediately
                loss, _, _, _, pxl_val, _ = criterion(recon, mask, mu, logvar, corrupted_input=corr)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            m['dice'] += d_acc.item(); m['pxl'] += pxl_val.item(); m['count'] += 1
            if i == 0 and (epoch + 1) % 10 == 0: save_visual_check(recon, mask, img, epoch+1, vis_dir, sid)
        rolling_dice = m['dice'] / m['count']
        print(f"Epoch {epoch+1:03d} | {current_phase} | Dice: {rolling_dice:.4f} | Pxl: {m['pxl']/m['count']:.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train_vae()