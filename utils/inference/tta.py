import torch
import torchvision.transforms.functional as TF

def get_tta_transforms():
    # Defines simple geometric transformations for TTA
    return ['identity', 'horizontal_flip', 'vertical_flip', 'rotate_90', 'rotate_180']


def tta_predict_10_augs(model, inputs, transforms_list):
    model.eval()
    predictions = []

    with torch.no_grad():
        for t in transforms_list:
            # Apply transformation
            if t == 'horizontal_flip':
                aug_input = TF.hflip(inputs)
            elif t == 'vertical_flip':
                aug_input = TF.vflip(inputs)
            elif t == 'rotate_90':
                aug_input = TF.rotate(inputs, 90)
            elif t == 'rotate_180':
                aug_input = TF.rotate(inputs, 180)  
            else:
                aug_input = inputs

            # Predict
            output = torch.sigmoid(model(aug_input))

            # Inverse transformation to align masks
            if t == 'horizontal_flip':
                output = TF.hflip(output)
            elif t == 'vertical_flip':
                output = TF.vflip(output)
            elif t == 'rotate_90':
                output = TF.rotate(output, -90)
            elif t == 'rotate_180':
                output = TF.rotate(inputs, 180)
            predictions.append(output)

    predictions_stack = torch.stack(predictions)
    predictions_max = predictions_stack.max(dim=1)[0]

    return predictions_stack, predictions_max