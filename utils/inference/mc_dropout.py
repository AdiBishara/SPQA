import torch
import torch.nn as nn


def monte_carlo_dropout_predict(model, inputs, num_samples=10):
    """Performs Monte Carlo Dropout inference, returning prediction stack and mean expected prediction."""
    model.eval() # Freeze BatchNorm stats

    # Force Dropout layers to Train mode (including 2D/3D variants)
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

    # Stack on dim=1: (Batch, Samples, Channel, H, W) for variance calculation
    predictions_stack = torch.stack(predictions, dim=1)

    # 4. Calculate Mean (Expected Prediction)
    predictions_mean = predictions_stack.mean(dim=1)

    # Return tuple to match evaluation.py expectation
    return predictions_stack, predictions_mean