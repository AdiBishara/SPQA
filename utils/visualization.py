import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def plot_and_save_images(test_images, test_annotations, target_np, output_np, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(test_images.cpu()[0][0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(test_annotations.cpu()[0], cmap='gray')
    axes[1].set_title('Original Label')
    axes[1].axis('off')

    axes[2].imshow(target_np.cpu(), cmap='gray')
    axes[2].set_title('Segmentation Output')
    axes[2].axis('off')

    axes[3].imshow(output_np, cmap='gray')
    axes[3].set_title('Shape Prior Label')
    axes[3].axis('off')

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base, ext = os.path.splitext(save_path)
    save_path = base + ".png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    mean_pred_folder = os.path.join(os.path.dirname(save_path), "Pseudo_label")
    os.makedirs(mean_pred_folder, exist_ok=True)
    mean_pred_fname = os.path.join(mean_pred_folder, os.path.basename(save_path))

    pred_img_uint8 = (output_np * 255).astype(np.uint8)
    im = Image.fromarray(pred_img_uint8, mode='L')
    im.save(mean_pred_fname)

def plot_segmentation_results_final(test_images, test_annotations, out, entropy, variance, pseudo, name, save_path):
    
    # Ensure the save path directory exists
    os.makedirs(save_path, exist_ok=True)

    # Extract filename (without extension)
    figname = name[0].split('.')[0]

    # Create folders for saving individual images
    folders = {
        "original": os.path.join(save_path, "original_image"),
        "ground_truth": os.path.join(save_path, "ground_truth"),
        "mean_pred": os.path.join(save_path, "mean_prediction"),
        "variance": os.path.join(save_path, "variance"),
        "pseudo_label": os.path.join(save_path, "pseudo_label")
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)  # Create each directory if it does not exist

    # Create a subplot with 1 row and 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Define image data and titles
    images_data = {
        "original": test_images.cpu()[0][0].numpy(),
        "ground_truth": test_annotations.cpu()[0].numpy(),
        "mean_pred": out["pred_mean_argmax"][0, 0],
        "variance": variance.cpu().numpy(),
        "pseudo_label": pseudo
    }

    titles = {
        "original": "Original Image",
        "ground_truth": "Ground Truth",
        "mean_pred": "Mean Model Predictions",
        "variance": "Model Uncertainty Variance",
        "pseudo_label": "Shape Prior Label"
    }

    colormaps = {
        "variance": "jet"  # Apply colormap for variance visualization
    }

    # Loop through each image type and save both plot and uint8 image
    for i, (key, image) in enumerate(images_data.items()):
        # Normalize and convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)

        # Save the image in its respective folder
        image_path = os.path.join(folders[key], f"{figname}.png")
        
        if key == "variance":  # Apply colormap to variance before saving
            plt.imsave(image_path, image, cmap=colormaps[key])
        else:
            Image.fromarray(img_uint8, mode='L').save(image_path)

        # Plot the image with the corresponding colormap
        if key == "variance":
            axes[i].imshow(image, cmap=colormaps[key])  # Use colormap for variance
        else:
            axes[i].imshow(img_uint8, cmap='gray')  # Normal grayscale images

        axes[i].set_title(titles[key], fontsize = 18)
        axes[i].axis('off')

    plt.tight_layout()

    # Save the full figure as a PDF
    pdf_path = os.path.join(save_path, f"{figname}.pdf")
    png_path = os.path.join(save_path, f"{figname}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    # Close the plot to free memory
    plt.close(fig)
