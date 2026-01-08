import torch
import torch.nn as nn


def monte_carlo_dropout_predict(model, inputs, num_samples=10):
    """
    Performs Monte Carlo Dropout inference.
    Returns:
        predictions_stack: (Batch, Samples, Channels, H, W)
        predictions_mean:  (Batch, Channels, H, W)
    """
    # 1. Set model to eval mode (freezes BatchNorm statistics)
    model.eval()

    # 2. Force Dropout layers to Train mode
    # FIX: Use isinstance to catch Dropout2d (which UNets use)
    def enable_dropout(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()

    model.apply(enable_dropout)

    predictions = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Forward pass (Dropout is active)
            output = torch.sigmoid(model(inputs))
            predictions.append(output)

    # 3. Stack predictions
    # FIX: Stack on dim=1 to get (Batch, Samples, Channel, H, W)
    # This ensures torch.var(dim=1) in evaluation.py works correctly.
    predictions_stack = torch.stack(predictions, dim=1)

    # 4. Calculate Mean (Expected Prediction)
    predictions_mean = predictions_stack.mean(dim=1)

    # Return tuple to match evaluation.py expectation
    return predictions_stack, predictions_mean