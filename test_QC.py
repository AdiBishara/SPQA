import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import pandas as pd
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.data.nifti_loader import NiftiFewShotDataset
from utils.model_loader import load_unet_model
from utils.models.dae import DAE
from utils.evaluation import compute_predictions_and_metrics

def run_quality_estimation():
    # 1. Load Configuration
    config_path = r"C:\Users\97252\SPQA\params\config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Trained Models
    # Load the Segmentor (UNet) using the helper
    unet, _ = load_unet_model(config['model'])
    
    # Load the QC Inspector (DAE)
    dae = DAE(
        in_channels=2, 
        out_channels=1,
        image_size=config['model']['image_size']
    ).to(device)
    dae_path = config['DAE']['checkpoint_path']
    
    if os.path.exists(dae_path):
        dae.load_state_dict(torch.load(dae_path, map_location=device))
        dae.eval()
        print(f"Loaded QC model from: {dae_path}")
    else:
        print(f"Warning: QC model checkpoint not found at {dae_path}")

    # 3. Setup Testing Data Loader
    # Uses the same NiftiFewShotDataset engine but with is_train=False
    test_dataset = NiftiFewShotDataset(
        data_root=config['Data']['raw_data_root'],
        id_file=config['Data']['test_ids'], 
        is_train=False,
        image_size=config['model']['image_size'][0]
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 4. Execute Inference & Metric Computation
    # This calls the MC Dropout/Uncertainty and DAE reconstruction logic
    print(f"Running quality estimation (Method: {config['method']})...")
    
    output_folder = r"C:\Users\97252\SPQA\results_plots"
    os.makedirs(output_folder, exist_ok=True)

    metrics_df = compute_predictions_and_metrics(
        model=unet,
        data_loader=test_loader,
        device=device,
        output_folder=output_folder,
        method=config['method'],
        model_dae=dae,
        model_dae_adv=None # Set this if using the GAN version
    )

    # 5. Save Results
    csv_save_path = os.path.join(output_folder, "quality_estimation_results.csv")
    metrics_df.to_csv(csv_save_path, index=False)
    print(f"Evaluation complete. CSV results saved to: {csv_save_path}")
    print(f"Visualizations generated in: {output_folder}")

if __name__ == "__main__":
    run_quality_estimation()