import torch


def deep_ensemble_predict(models, inputs):
    # 'models' is expected to be a list of pre-trained models
    predictions = []

    with torch.no_grad():
        for model in models:
            model.eval()
            output = torch.sigmoid(model(inputs))
            predictions.append(output)

    predictions_stack = torch.stack(predictions)
    predictions_max = predictions_stack.max(dim=1)[0]

    return predictions_stack, predictions_max