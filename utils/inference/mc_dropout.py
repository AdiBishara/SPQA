import torch


def monte_carlo_dropout_predict(model, inputs, num_samples=10):
    model.eval()  # Keep dropout layers active
    predictions = []

    with torch.no_grad():
        for _ in range(num_samples):
            output = torch.sigmoid(model(inputs))
            predictions.append(output)

    # Stack samples: (num_samples, batch, channel, h, w)
    predictions_stack = torch.stack(predictions)
    # Get max probabilities for uncertainty calculation
    predictions_max = predictions_stack.max(dim=1)[0]

    return predictions_stack, predictions_max