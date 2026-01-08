import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from torch.utils.data import DataLoader
import torch.nn.functional as F


# --- METRICS & HELPERS ---

def dice_coeff_metric(pred, target):
    """Computes the Dice Coefficient."""
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def compute_uncertainty_map(predictions: torch.Tensor):
    """Calculates pixel-wise uncertainty (Variance)."""
    # predictions: (Batch, Samples, Channels, H, W)
    # Variance across dim=1 (Samples)
    return torch.var(predictions, dim=1)


def compute_uncertainty_scalar(uncertainty_map: torch.Tensor):
    return uncertainty_map.mean().item()


def force_dropout_on(model):
    """Finds all dropout layers and forces them to TRAIN mode."""
    count = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()
            count += 1
    return count


# -------------------------

def evaluate_segmentation_quality(
        prediction: torch.Tensor,
        target: torch.Tensor,
        uncertainty_map: torch.Tensor,
        uncertainty_scalar: float,
        dae_model: Optional[torch.nn.Module] = None,
        dae_adv_model: Optional[torch.nn.Module] = None
) -> Dict[str, float]:
    metrics = {}

    pred_binary = (prediction > 0.5).float()
    target_binary = (target > 0.5).float()

    metrics['dice_score'] = dice_coeff_metric(pred_binary, target_binary).item()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    metrics['iou'] = (intersection / (union + 1e-6)).item()

    metrics['uncertainty_scalar'] = uncertainty_scalar
    metrics['uncertainty_mean'] = uncertainty_map.mean().item()
    metrics['uncertainty_max'] = uncertainty_map.max().item()

    return metrics


def save_visualization(image, target, prediction, uncertainty, save_path, subject_id, metrics):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f"Image: {subject_id}")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(target.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(prediction.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f"Pred (Dice: {metrics['dice_score']:.2f})")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(uncertainty.squeeze().cpu().numpy(), cmap='jet')
    plt.title(f"Uncertainty: {metrics['uncertainty_scalar']:.3f}")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_predictions_and_metrics(
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        output_folder: str,
        method: str = "mc_dropout",
        model_dae: Optional[torch.nn.Module] = None,
        model_dae_adv: Optional[torch.nn.Module] = None
) -> pd.DataFrame:
    # 1. Start with Eval (locks BatchNorm)
    model.eval()
    if model_dae: model_dae.eval()
    if model_dae_adv: model_dae_adv.eval()

    results = []
    print(f"Starting inference on {len(data_loader)} slices...")

    # Flag to print debug info only once
    debug_printed = False

    for i, (test_images, test_masks, subject_ids) in enumerate(data_loader):
        test_images = test_images.to(device)
        test_masks = test_masks.to(device)

        # Ensure 4D (Batch, Channel, H, W)
        if len(test_images.shape) == 3:
            test_images = test_images.unsqueeze(1)

        # --- MC DROPOUT INFERENCE ---
        if method == "mc_dropout":

            # A. FORCE DROPOUT ON
            active_layers = force_dropout_on(model)

            # B. PARANOID CHECK (Print status once)
            if not debug_printed:
                # Grab the first dropout layer to verify
                for m in model.modules():
                    if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
                        print(f"DEBUG CHECK: Dropout Layer Mode is TRAIN? -> {m.training}")
                        break
                print(f"DEBUG CHECK: Forced {active_layers} layers to Train Mode.")
                debug_printed = True

            # C. RUN SAMPLES
            mc_outputs = []
            num_samples = 10

            with torch.no_grad():
                for _ in range(num_samples):
                    logits = model(test_images)
                    probs = torch.sigmoid(logits)
                    mc_outputs.append(probs)

            # Stack (Batch, Samples, C, H, W)
            predictions = torch.stack(mc_outputs, dim=1)

            # D. CALCULATE METRICS
            uncertainty_map = compute_uncertainty_map(predictions)
            uncertainty_scalar = compute_uncertainty_scalar(uncertainty_map)
            final_pred = (predictions.mean(dim=1) > 0.5).float()

        else:
            with torch.no_grad():
                output = torch.sigmoid(model(test_images))
            final_pred = (output > 0.5).float()
            uncertainty_map = torch.zeros_like(final_pred)
            uncertainty_scalar = 0.0

        # --- QC CHECKS ---
        rec_error_dae = 0.0
        if model_dae:
            with torch.no_grad():
                dae_input = torch.cat([test_images, final_pred], dim=1)
                reconstruction = torch.sigmoid(model_dae(dae_input))
                rec_error_dae = F.mse_loss(reconstruction, final_pred).item()

        rec_error_gan = 0.0
        if model_dae_adv:
            with torch.no_grad():
                dae_input = torch.cat([test_images, final_pred], dim=1)
                reconstruction_adv = torch.sigmoid(model_dae_adv(dae_input))
                rec_error_gan = F.mse_loss(reconstruction_adv, final_pred).item()

        # --- SAVE RESULTS ---
        for b in range(test_images.size(0)):
            subj_id = subject_ids[b] if isinstance(subject_ids, list) else subject_ids

            current_metrics = evaluate_segmentation_quality(
                final_pred[b],
                test_masks[b],
                uncertainty_map[b],
                uncertainty_scalar
            )

            current_metrics['rec_error_dae'] = rec_error_dae
            current_metrics['rec_error_gan'] = rec_error_gan
            current_metrics['subject_id'] = subj_id

            results.append(current_metrics)

            save_path = os.path.join(output_folder, f"{subj_id}_result.png")
            save_visualization(
                test_images[b],
                test_masks[b],
                final_pred[b],
                uncertainty_map[b],
                save_path,
                subj_id,
                current_metrics
            )

    df = pd.DataFrame(results)
    return df