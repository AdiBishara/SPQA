import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.model_loader import load_unet_model
from utils.models.dae import DAE
from utils.evaluation import compute_predictions_and_metrics
import glob


def get_next_run_path(base_dir, subject_name):
    os.makedirs(base_dir, exist_ok=True)
    search_pattern = os.path.join(base_dir, f"{subject_name}_Run_*")
    existing_folders = glob.glob(search_pattern)
    max_run = 0
    for folder in existing_folders:
        try:
            folder_name = os.path.basename(folder)
            run_num = int(folder_name.split('_')[-1])
            if run_num > max_run: max_run = run_num
        except ValueError:
            continue
    return os.path.join(base_dir, f"{subject_name}_Run_{max_run + 1}")


def run_quality_estimation():
    # 1. Load Config
    config_path = r"C:\Users\97252\SPQA\params\config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- FORCE SETTINGS FOR QC ---
    config['method'] = 'mc_dropout'  # <--- FORCE UNCERTAINTY ON
    # -----------------------------

    # 2. Load Segmentation Model
    unet, _ = load_unet_model(config['model'])

    # 3. Load QC Model 1 (Standard DAE)
    print("Loading Standard QC Model (DAE)...")
    dae = DAE(in_channels=2, out_channels=1, image_size=config['model']['image_size']).to(device)
    # POINT THIS TO YOUR LATEST EPOCH 100 CHECKPOINT
    dae_path = r"C:\Users\97252\SPQA\logs\ae_checkpoints\dae_normal_epoch_100.pth"
    if os.path.exists(dae_path):
        dae.load_state_dict(torch.load(dae_path, map_location=device))
        dae.eval()
    else:
        print(f"⚠️ WARNING: DAE Checkpoint not found at {dae_path}")
        dae = None

    # 4. Load QC Model 2 (Adversarial GAN) - THIS WAS MISSING
    print("Loading Adversarial QC Model (GAN)...")
    dae_adv = DAE(in_channels=2, out_channels=1, image_size=config['model']['image_size']).to(device)
    # POINT THIS TO YOUR LATEST EPOCH 100 CHECKPOINT
    gan_path = r"C:\Users\97252\SPQA\logs\gan_checkpoints\dae_gan_epoch_100.pth"
    if os.path.exists(gan_path):
        dae_adv.load_state_dict(torch.load(gan_path, map_location=device))
        dae_adv.eval()
    else:
        print(f"⚠️ WARNING: GAN Checkpoint not found at {gan_path}")
        dae_adv = None

    # 5. Setup Data
    test_dataset = NiftiFewShotDataset(
        data_root=config['Data']['raw_data_root'],
        id_file=config['Data']['test_ids'],
        is_train=False,
        image_size=config['model']['image_size'][0]
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 6. Output Setup
    base_output_folder = r"C:\Users\97252\SPQA\results_plots"
    try:
        with open(config['Data']['test_ids'], 'r') as f:
            subject_name = f.readline().strip()
    except:
        subject_name = "UnknownSubject"

    current_run_folder = get_next_run_path(base_output_folder, subject_name)
    os.makedirs(current_run_folder, exist_ok=True)
    print(f"--> Saving results to: {current_run_folder}")

    # 7. Run Evaluation
    print(f"Running quality estimation (Method: {config['method']})...")
    metrics_df = compute_predictions_and_metrics(
        model=unet,
        data_loader=test_loader,
        device=device,
        output_folder=current_run_folder,
        method=config['method'],
        model_dae=dae,
        model_dae_adv=dae_adv  # <--- Now passing the GAN model
    )

    # 8. Save
    csv_save_path = os.path.join(current_run_folder, "quality_estimation_results.csv")
    metrics_df.to_csv(csv_save_path, index=False)
    print(f"Done. Results saved to: {csv_save_path}")


if __name__ == "__main__":
    run_quality_estimation()