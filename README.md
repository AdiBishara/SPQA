# SPQA: Segmentation Pseudo-label Quality Assurance

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Abstract:** Automated quality control of medical image segmentation is a critical bottleneck in deploying deep learning models to clinical workflows. This repository presents **SPQA**, a novel framework that leverages 3D Denoising Autoencoders (DAEs) to independently evaluate the quality of segmentation pseudo-labels. By directly modeling the anatomical shape priors of brain structures (e.g., using the SynthStrip dataset), the DAE reconstructs "clean" masks from potentially flawed segmentations. The discrepancy between the input segmentation and the DAE reconstruction serves as a highly reliable, reference-free quality score, correlating strongly with traditional volumetric overlap metrics (Dice).

---

## 📖 Table of Contents
1. [Methodology](#methodology)
2. [Repository Architecture](#repository-architecture)
3. [Getting Started](#getting-started)
4. [Dataset Preparation](#dataset-preparation)
5. [Reproducing the Pipeline](#reproducing-the-pipeline)
6. [Citation & Contact](#citation--contact)

---

## 🧠 Methodology

The SPQA pipeline is decoupled into two primary components to ensure the quality assurance mechanism remains independent of the segmentation model's internal biases.

1. **Primary Segmentation (U-Net):** A robust 3D U-Net architecture is employed for the base segmentation task. To establish baseline uncertainty, this model is evaluated using Test-Time Augmentation (TTA), Deep Ensembles, and Monte Carlo Dropout (MC-Dropout).
2. **Quality Control Evaluators:**
   * **Denoising Autoencoder (DAE):** A structurally optimized 3D U-Net serving as the core of SPQA. It is trained to map artificially corrupted segmentations (alongside the original anatomical MRI) back to the pristine ground truth. At inference time, high reconstruction error strongly indicates that the primary U-Net produced an anatomically implausible segmentation.
   * **Variational Autoencoder (VAE):** We provide a custom 3D VAE with a preserved spatial bottleneck (avoiding 1D flattening) as a probabilistic baseline to model the distribution of healthy brain morphology.

### Loss Formulation
The SPQA DAE is trained using a dynamic, curriculum-driven composite loss function (`DAELoss`):
* **Binary Cross-Entropy (BCE):** With dynamic band-masking to focus on tissue boundaries.
* **Regional Dice Loss:** To ensure global volumetric overlap.
* **Active Contour Loss:** Explicitly rewarding high-frequency alignment of sulci/gyri edges.

---

## 📂 Repository Architecture

The codebase is organized modularly to support reproducibility and extension.

```text
SPQA/
├── losses/
│   └── losses.py                # Implementation of DAELoss, boundary contour, and Dice metrics
├── params/
│   └── config.yaml              # Centralized hyperparameters and phased-training curriculum
├── utils/
│   ├── data/
│   │   └── nifti_loader.py      # Dataloaders for 3D NIfTI volumes (SynthStrip)
│   ├── inference/
│   │   ├── ensemble.py          # Deep Ensemble inference utilities 
│   │   ├── mc_dropout.py        # MC-Dropout epistemic uncertainty estimation
│   │   └── tta.py               # Test-Time Augmentation protocols
│   ├── models/
│   │   ├── unet_dae.py          # 3D U-Net configured as a Denoising Autoencoder
│   │   ├── unet_dropout.py      # Primary 3D U-Net with native MC-Dropout support
│   │   └── vae.py               # Variational Autoencoder prototype (spatial bottleneck)
│   ├── config.py                # YAML parser mapping
│   ├── evaluation.py            # Post-inference statistical evaluation functions
│   ├── model_loader.py          # Device-agnostic checkpoint loading
│   ├── seeding.py               # Strict deterministic seeding (cuDNN, NumPy, Torch)
│   └── visualization.py         # Subplot generation for NIfTI slices
├── train_segmentation.py        # Entry point: Trains the primary U-Net
├── train_QC_AE.py               # Entry point: Trains the Quality Control DAE
├── train_VAE.py                 # Entry point: Trains the VAE comparative baseline
├── run_QC.py                    # Inference: Evaluates U-Net segmentation performance
├── run_VAE_eval.py              # Inference: Evaluates DAE/VAE reconstruction quality
└── cp_selector.py               # Utility: Interactively compare validation checkpoints
```

---

## 🚀 Getting Started

### Prerequisites
* **OS:** Linux / Windows
* **Hardware:** CUDA-capable GPU with at least 12GB VRAM (24GB+ recommended for 3D volumes)
* **Python:** 3.9 or higher

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/AdiBishara/SPQA.git
cd SPQA
pip install -r requirements.txt
```

*(Note: We highly recommend using a virtual environment such as `conda` or `venv`)*

---

## 💽 Dataset Preparation

This project natively supports the **SynthStrip v1.5** brain extraction dataset. 

1. Download the SynthStrip dataset.
2. Structure the data directory strictly as follows:

```text
synthstrip_data_v1.5/
├── subject_001/
│   ├── image.nii.gz          # Raw T1/T2 MRI Volume
│   ├── pseudo_label.nii.gz   # Simulated or generated segmentation
│   └── truth.nii.gz          # Ground Truth mask
├── subject_002/
...
```

3. Update the `raw_data_root` variable inside `params/config.yaml` to point to your `synthstrip_data_v1.5/` directory.

---

## 🔬 Reproducing the Pipeline

The framework is strictly driven by the hyperparameters configured in `params/config.yaml`. Adjust batch sizes according to your hardware limits before running the scripts below.

### 1. Primary Segmentation Training
Train the foundational U-Net model on the dataset to generate baseline predictions:
```bash
python train_segmentation.py
```

### 2. Quality Control (DAE) Training
Train the structural Denoising Autoencoder to learn brain shape priors:
```bash
python train_QC_AE.py
```

### 3. VAE Baseline Training (Optional)
To train the probabilistic Variational Autoencoder baseline for comparative distribution modeling:
```bash
python train_VAE.py
```

### 4. Evaluate Segmentation & Uncertainty
Run inference using the trained U-Net. This script natively executes MC-Dropout to generate dense prediction bounds and estimates Dice scores:
```bash
python run_QC.py
```

### 5. Evaluate Reconstruction Quality (DAE / VAE)
Measure the ability of the autoencoders to identify anomalies. The output correlates the reconstruction error against the true Dice score:
```bash
# Evaluate DAE
python validate_dae.py

# Evaluate VAE
python validate_vae.py

# Consolidated Evaluation
python run_VAE_eval.py
```

### 6. Validation Checkpoint Selection
Compare various epoch checkpoints interactively to track training convergence:
```bash
python cp_selector.py
```

---

## 📄 Citation & Contact

If you utilize this codebase or framework for your research, please cite our corresponding paper:

*(Citation details will be updated pending publication)*

For academic inquiries, collaboration, or access to pre-trained weights, please open an Issue in this repository or contact the authors directly.

---
*Developed as part of academic research into robust clinical deep learning deployment.*
