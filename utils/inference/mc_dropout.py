import torch
import torch.nn as nn


def monte_carlo_dropout_predict(model, inputs, num_samples=10):
    # 1. Set model to eval mode (freezes BatchNorm)
    model.eval()

    def enable_dropout(m):
        if type(m) == nn.Dropout:
            m.train()

    model.apply(enable_dropout)

    predictions = []

    with torch.no_grad():
        for _ in range(num_samples):
            output = torch.sigmoid(model(inputs))
            predictions.append(output)

    # Shape: (num_samples, batch, channel, h, w)
    predictions_stack = torch.stack(predictions)

    # usually you return the stack, or the mean/std across dim=0 (samples)
    return predictions_stack