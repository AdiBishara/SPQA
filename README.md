# SPQA — Segmentation Pseudo-label Quality Assurance

Deep learning framework for automated quality control of brain segmentation using a 3D Denoising Autoencoder (DAE). The DAE reconstructs clean masks from corrupted pseudo-labels; higher reconstruction errors indicate lower quality pseudo-labels. Built on [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/).

---

## Architecture

| Component | Role | File |
|---|---|---|
| **DAE (via VAE3D)** | Reconstruction-based quality estimator (2-channel input: image + corrupted PL &rarr; 1-channel reconstruction) | `utils/models/vae.py` |
| **U-Net** | Primary segmentation model (with MC-Dropout for uncertainty) | `utils/models/unet_dropout.py` |
| **DAELoss** | Composite loss: BCE (dynamic band) + Dice + Contour + KLD | `losses/losses.py` |
| **Phased Training** | Config-driven curriculum that shifts loss weights as Dice improves (volume &rarr; boundary focus) | `params/config.yaml` |

## Project Structure

```
SPQA/
├── params/
│   └── config.yaml              # Central config (model, training, loss phases)
├── utils/
│   ├── models/
│   │   ├── vae.py               # VAE3D architecture (3D encoder-decoder)
│   │   └── unet_dropout.py      # U-Net wrapper with MC-Dropout
│   ├── data/
│   │   └── nifti_loader.py      # NIfTI dataset loaders
│   ├── config.py                # YAML config loader
│   ├── seeding.py               # Reproducibility utilities
│   ├── evaluation.py            # Evaluation helpers
│   ├── visualization.py         # Plotting utilities
│   └── model_loader.py          # Model loading helpers
├── losses/
│   └── losses.py                # DAELoss, Dice, Boundary contour distances
├── train_QC_AE.py               # Train the DAE (quality control autoencoder)
├── train_segmentation.py        # Train the U-Net segmentation model
├── run_QC.py                    # Evaluate U-Net segmentation (Dice, HD95, IoU)
├── run_VAE_eval.py              # Evaluate DAE reconstruction quality
├── validator.py                 # Cross-fold validation of VAE quality scores
├── hallucinations.py            # Anomaly detection test (synthetic artifact injection)
├── cp_selector.py               # Checkpoint comparison tool
└── fold_4/                      # Data split IDs (training/test)
```

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, 256³ volumes are memory-intensive)

### Installation

```bash
git clone https://github.com/AdiBishara/SPQA.git
cd SPQA
pip install -r requirements.txt
```

### Data

Place [SynthStrip v1.5](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) data under a `synthstrip_data_v1.5/` directory. Each subject should have:
```
synthstrip_data_v1.5/<subject_id>/
├── image.nii.gz
├── pseudo_label.nii.gz
└── truth.nii.gz
```

Update the `data.raw_data_root` path in `params/config.yaml` to point to your data directory.

## Usage

### 1. Train the Segmentation U-Net

```bash
python train_segmentation.py
```

### 2. Train the VAE (Quality Control Model)

```bash
python train_QC_AE.py
```

### 3. Evaluate Segmentation Quality

```bash
python run_QC.py
```

### 4. Evaluate VAE Reconstruction

```bash
python run_VAE_eval.py
```

### 5. Cross-fold Validation

```bash
# Use the latest checkpoint automatically
python validator.py

# Specify a checkpoint
python validator.py --checkpoint logs/best_cps/run_22_best_cps/vae_epoch_733.pth

# Specify latent dimension if checkpoint was trained with different dim
python validator.py --checkpoint path/to/model.pth --latent-dim 4096
```

### 6. Anomaly Detection Test

```bash
python hallucinations.py
```

## Configuration

All training hyperparameters are in `params/config.yaml`. Key sections:

- **`model`** — Architecture settings (channels, latent dim, image size)
- **`train`** — Learning rate, epochs, batch size, and `kld_weight` (which controls deterministic DAE vs probabilistic VAE)
- **`phases`** — Loss weight curriculum (automatically transitions as Dice improves):
  - Transitions blend `dice`, `contour`, `bce`, and `band_size` to progressively heavily penalize contour/boundary errors once volumetric overlap is achieved.

## License

This project is part of academic research. Please contact the authors for licensing information.
