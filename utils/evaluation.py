import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_erosion
import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from medpy import metric
import pickle
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from utils.visualization import plot_and_save_images, plot_segmentation_results_final
from utils.transformations import forward_transform, inverse_transform
from utils.inference import tta, ensemble, mc_dropout
from utils.seeding import fix_seeds
import cv2
import matplotlib.pyplot as plt
import nibabel as nib # Ensure nibabel is imported

# Set global seed
fix_seeds(42)

def calculate_metric_perslice_fullhd(pred, gt):
    smooth = 1e-6
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    dice =  (2.0 * intersection + smooth) / (union + smooth)

    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        myhd = metric.binary.hd(pred, gt)
        return dice, myhd , pred.sum() , gt.sum()
    elif pred.sum() == 0 and gt.sum() == 0:
        return dice, 0, pred.sum(), gt.sum()
    else:
        return dice, 0 , pred.sum() , gt.sum()

def dice_coefficient_np(prediction, target, smooth=1e-6):
    intersection = np.sum(prediction * target)
    union = np.sum(prediction) + np.sum(target)
    return (2.0 * intersection + smooth) / (union + smooth), np.sum(prediction), np.sum(target)

def dice_coefficient(y_true, y_pred, epsilon=1e-7):
    intersection = np.sum(y_true * y_pred)
    sum_masks = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + epsilon) / (sum_masks + epsilon)

def hausdorff_distance(y_true, y_pred):
    coords_true = np.argwhere(y_true)
    coords_pred = np.argwhere(y_pred)

    if coords_true.size == 0 or coords_pred.size == 0:
        return np.inf

    forward_hd = directed_hausdorff(coords_true, coords_pred)[0]
    backward_hd = directed_hausdorff(coords_pred, coords_true)[0]
    return max(forward_hd, backward_hd)

def binary_cross_entropy(target, output, epsilon=1e-7):
    output = np.clip(output, epsilon, 1 - epsilon)
    bce = -(target * np.log(output) + (1 - target) * np.log(1 - output))
    return np.mean(bce)

def hamming_distance(target, output):
    return np.mean(target != output)

def compute_uncertainty(class_probabilities, class_probabilities_max):
    mean_class_probabilities = torch.mean(class_probabilities, dim=0)
    mean_class_probabilities_max = torch.mean(class_probabilities_max.float(), dim=0)

    entropy = -torch.sum(mean_class_probabilities * torch.log(mean_class_probabilities + 1e-9), dim=0)
    variance = torch.mean(class_probabilities * (1 - class_probabilities), dim=0).sum(dim=0)

    mean_entropy = torch.mean(entropy)
    mean_variance = torch.mean(variance)

    mean_prediction_map = torch.mean(mean_class_probabilities, dim=0)

    normalized_entropy = entropy / (mean_prediction_map + 1e-6)
    normalized_variance = variance / (mean_prediction_map + 1e-6)

    normalized_entropy_area = normalized_entropy.sum()
    normalized_variance_area = normalized_variance.sum()

    total_pixels = entropy.numel()
    normalized_entropy_fraction = normalized_entropy_area / total_pixels
    normalized_variance_fraction = normalized_variance_area / total_pixels

    return (class_probabilities, mean_class_probabilities, entropy, variance,
            mean_entropy, mean_variance,
            normalized_entropy_area, normalized_entropy_fraction,
            normalized_variance_area, normalized_variance_fraction)

def compute_dice_and_hd(pred_samples, groundtruth):
    num_samples = len(pred_samples)
    pred_mean = np.mean(pred_samples, axis=0, keepdims=True) 

    pred_mean = torch.from_numpy(pred_mean)
    pred_mean_argmax = pred_mean.max(1, keepdim=True)[1]
    pred_mean_argmax = pred_mean_argmax.numpy()
    dice_coefficients = []
    hausdorff_distances = []
    
    for samp_no in range(num_samples):
        sample_torch = torch.from_numpy(np.expand_dims(pred_samples[samp_no], axis=0))
        sample_torch_argmax = sample_torch.max(1, keepdim=True)[1]
        sample_torch_argmax = sample_torch_argmax.numpy()
        
        dice,_,_ = dice_coefficient_np(sample_torch_argmax[0,0], pred_mean_argmax[0,0])
        dice_coefficients.append(dice)

        _, hd , pred_sum , gt_sum = calculate_metric_perslice_fullhd(sample_torch_argmax, pred_mean_argmax)
        hausdorff_distances.append(hd)

    dice_coefficients = np.array(dice_coefficients)
    hausdorff_distances = np.array(hausdorff_distances)
    gt_dice, pred_sum_f , gt_sum_f  = dice_coefficient_np(pred_mean_argmax[0,0], groundtruth.cpu().numpy())
    
    def compute_statistics(values):
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)
        var = np.var(values)
        minimum = np.min(values)
        maximum = np.max(values)
        return {
            "mean": mean, "median": median, "std": std, "variance": var, "min": minimum, "max": maximum,
        }

    dice_stats = compute_statistics(dice_coefficients)
    hd_stats = compute_statistics(hausdorff_distances)
    
    return {
        "Dice_Statistics": dice_stats,
        "Hausdorff_Distance_Statistics": hd_stats,
        "GT_Dice": gt_dice,
        "pred_sum_f": pred_sum_f,
        "pred_mean_argmax": pred_mean_argmax
    }

def compute_dice(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    dice = 2. * intersection / (mask1.sum() + mask2.sum())
    return dice 
    
def compute_hd(mask1, mask2):
    coords1 = np.argwhere(mask1)
    coords2 = np.argwhere(mask2)
    if len(coords1) == 0 or len(coords2) == 0:
        return 0.0, 0.0
    hd_1to2 = directed_hausdorff(coords1, coords2)[0]
    hd_2to1 = directed_hausdorff(coords2, coords1)[0]
    return hd_1to2, hd_2to1

def compute_chamfer_distance(mask1, mask2):
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    dist1 = cv2.distanceTransform(1 - mask2, cv2.DIST_L2, 3)
    dist2 = cv2.distanceTransform(1 - mask1, cv2.DIST_L2, 3)
    chamfer_1to2 = dist1[mask1.astype(bool)].mean() if mask1.sum() > 0 else 0
    chamfer_2to1 = dist2[mask2.astype(bool)].mean() if mask2.sum() > 0 else 0
    return chamfer_1to2 , chamfer_2to1    

def compute_predictions_and_metrics(model, data_loader, device, output_folder, method, model_dae, model_dae_adv):
    metrics_data = []
    current_vol_data = []
    current_subject = None
    
    with torch.no_grad():
        pbar = tqdm(enumerate(BackgroundGenerator(data_loader)), total=len(data_loader))
        for itr, (test_images, test_annotations, name) in pbar:
            subject_id = name[0].split('_slice_')[0]
            test_images = test_images.to(device=device, dtype=torch.float32).squeeze(1)
            test_annotations = test_annotations.to(device=device, dtype=torch.long).squeeze(1)

            if method == "MC":
                predictions, predictions_max = mc_dropout.monte_carlo_dropout_predict(model, test_images, num_samples=10)
            elif method == "deep_ensemble":
                predictions, predictions_max = ensemble.deep_ensemble_predict(model, test_images)
            elif method == "TTA":
                tta_transforms = tta.get_tta_transforms()
                predictions, predictions_max = tta.tta_predict_10_augs(model, test_images, tta_transforms)

            (class_probabilities, mean_class_probabilities, entropy, variance,
             mean_entropy, mean_variance,
             normalized_entropy_area, normalized_entropy_fraction,
             normalized_variance_area, normalized_variance_fraction) = compute_uncertainty(predictions, predictions_max)

            out = compute_dice_and_hd(class_probabilities.cpu().numpy(), test_annotations[0])
            pred_array = out['pred_mean_argmax'][0, 0].astype(np.float32)
            pred_tensor = torch.from_numpy(pred_array).to(device=device)
            
            # Prepare Input for DAE: Stack [Image + Mask]
            dae_input = torch.cat([
                forward_transform(test_images).unsqueeze(0), 
                forward_transform(pred_tensor.unsqueeze(0).unsqueeze(0))
            ], dim=1)
            
            # FIX: Feed dae_input (2 channels) instead of transformed_img (1 channel)
            outputs_dae = torch.sigmoid(model_dae(dae_input))

            target_np = pred_tensor.cpu().numpy().astype(np.uint8)
            output_np_dae = outputs_dae[0, 0].cpu().numpy()
            
            # Visualize the result before inverse transform
            output_np_dae_bin = (output_np_dae > 0.5).astype(np.uint8)

            # Inverse transform to map back to original size for metric calculation
            pred_tensor2 = torch.from_numpy(output_np_dae_bin).to(device=device)
            output_np_gt_dae = inverse_transform(pred_tensor2.unsqueeze(0).unsqueeze(0)).cpu().numpy()
            output_np_gt_dae = (output_np_gt_dae[0][0] > 0.5).astype(np.uint8)
            target_np = (target_np > 0.5).astype(np.uint8)

            dice_score_val_dae = dice_coefficient(target_np, output_np_gt_dae)
            hausdorff_dist_dae = hausdorff_distance(target_np, output_np_gt_dae)

            plot_segmentation_results_final(
                test_images, test_annotations, out, entropy, variance, output_np_gt_dae, name, output_folder
            )
            
            dice_score = compute_dice(target_np, output_np_gt_dae)
            hd1, hd2 = compute_hd(target_np, output_np_gt_dae)
            chamfer1, chamfer2 = compute_chamfer_distance(target_np, output_np_gt_dae)
            
            metrics_row = {
                'Image_Name': name[0],
                'mean_entropy': mean_entropy.item(),
                'mean_variance': mean_variance.item(),
                'normalized_entropy_area': normalized_entropy_area.item(),
                'normalized_entropy_fraction': normalized_entropy_fraction.item(),
                'normalized_variance_area': normalized_variance_area.item(),
                'normalized_variance_fraction': normalized_variance_fraction.item(),
                'dice_mean': out["Dice_Statistics"]["mean"],
                'dice_max': out["Dice_Statistics"]["max"],
                'dice_min': out["Dice_Statistics"]["min"],
                'dice_std': out["Dice_Statistics"]["std"],
                'dice_variance': out["Dice_Statistics"]["variance"],
                'hd_mean': out["Hausdorff_Distance_Statistics"]["mean"],
                'hd_max': out["Hausdorff_Distance_Statistics"]["max"],
                'hd_min': out["Hausdorff_Distance_Statistics"]["min"],
                'hd_std': out["Hausdorff_Distance_Statistics"]["std"],
                'hd_variance': out["Hausdorff_Distance_Statistics"]["variance"],
                'gt_dice': out['GT_Dice'],
                'mean_pred_sum': out['pred_sum_f'],
                "Dice_DAE": dice_score_val_dae,
                "Hausdorff_DAE": hausdorff_dist_dae,
                'Dice_D': dice_score,
                'HD1_D': hd1,
                'HD2_D': hd2,
                'CHD1_D':chamfer1,
                'CHD2_D':chamfer2,
            }
            
            # --- 3D Reconstruction Logic ---
            # Save the CLEANED slice (inverse transformed to match original size)
            cleaned_slice_final = output_np_gt_dae 
            
            if current_subject is not None and subject_id != current_subject:
                # Save previous volume
                if current_vol_data:
                    vol_3d = np.stack(current_vol_data, axis=2)
                    nifti_img = nib.Nifti1Image(vol_3d, affine=np.eye(4))
                    nib.save(nifti_img, os.path.join(output_folder, f"{current_subject}_reconstructed.nii.gz"))
                    print(f"Saved 3D volume for {current_subject}")
                current_vol_data = [] # Reset

            current_subject = subject_id
            current_vol_data.append(cleaned_slice_final)
            metrics_data.append(metrics_row)
            
        # Save the last volume after loop ends
        if current_subject is not None and current_vol_data:
            vol_3d = np.stack(current_vol_data, axis=2)
            nifti_img = nib.Nifti1Image(vol_3d, affine=np.eye(4))
            nib.save(nifti_img, os.path.join(output_folder, f"{current_subject}_reconstructed.nii.gz"))    
    
    return pd.DataFrame(metrics_data)