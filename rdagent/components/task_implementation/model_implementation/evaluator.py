import torch
import numpy as np


def shape_evaluator(target, prediction):
    if target is None or prediction is None:
        return None, 0
    tar_shape = target.shape
    pre_shape = prediction.shape

    diff = []
    for i in range(max(len(tar_shape), len(pre_shape))):
        dim_tar = tar_shape[i] if i < len(tar_shape) else 0
        dim_pre = pre_shape[i] if i < len(pre_shape) else 0
        diff.append(abs(dim_tar - dim_pre))

    metric = 1 / (np.exp(np.mean(diff)) + 1)
    return diff, metric


def reshape_tensor(original_tensor, target_shape):
    new_tensor = torch.zeros(target_shape)
    for i, dim in enumerate(original_tensor.shape):
        new_tensor = new_tensor.narrow(i, 0, dim).copy_(original_tensor)

    return new_tensor


def value_evaluator(target, prediction):
    if target is None or prediction is None:
        return None, 0
    tar_shape = target.shape
    pre_shape = prediction.shape

    # Determine the shape of the padded tensors
    dims = [
        max(s1, s2)
        for s1, s2 in zip(
            tar_shape + (1,) * (len(pre_shape) - len(tar_shape)),
            pre_shape + (1,) * (len(tar_shape) - len(pre_shape)),
        )
    ]
    # Reshape both tensors to the determined shape
    target = target.reshape(
        *tar_shape, *(1,) * (max(len(tar_shape), len(pre_shape)) - len(tar_shape))
    )
    prediction = prediction.reshape(
        *pre_shape, *(1,) * (max(len(tar_shape), len(pre_shape)) - len(pre_shape))
    )
    target_padded = reshape_tensor(target, dims)
    prediction_padded = reshape_tensor(prediction, dims)

    # Calculate the mean absolute difference
    diff = torch.abs(target_padded - prediction_padded)
    metric = 1 / (1 + np.exp(torch.mean(diff).item()))
    return diff, metric


if __name__ == "__main__":
    tar = torch.rand(4, 5, 5)
    pre = torch.rand(4, 1)
    print(shape_evaluator(tar, pre))
    print(value_evaluator(tar, pre)[1])
