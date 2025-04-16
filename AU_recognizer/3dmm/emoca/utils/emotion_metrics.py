# THIS FILE IS BORROWED FROM https://github.com/face-analysis/emonet/
# TORCH FUNCTIONS
import torch


def ACC_torch(ground_truth, predictions):
    """Evaluates the mean accuracy
    """
    assert ground_truth.shape == predictions.shape
    return torch.mean(torch.eq(ground_truth.int(), predictions.int()).float())


def SAGR_torch(ground_truth, predictions):
    """
        Evaluates the SAGR between estimate and ground truth.
    """
    assert ground_truth.shape == predictions.shape
    return torch.mean(torch.eq(torch.sign(ground_truth), torch.sign(predictions)).float())


def PCC_torch(ground_truth, predictions, batch_first=True, weights=None):
    """
        Evaluates the Pearson Correlation Coefficient.
        Inputs are numpy arrays.
        Corr = Cov(GT, Est)/(std(GT)std(Est))
    """
    assert ground_truth.shape == predictions.shape
    assert predictions.numel() >= 2  # std doesn't make sense, unless there is at least two items in the batch
    if batch_first:
        dim = -1
    else:
        dim = 0
    if weights is None:
        centered_x = ground_truth - ground_truth.mean(dim=dim, keepdim=True)
        centered_y = predictions - predictions.mean(dim=dim, keepdim=True)
        covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
        x_std = ground_truth.std(dim=dim, keepdim=True)
        y_std = predictions.std(dim=dim, keepdim=True)
    else:
        # weighted average
        weights = weights / weights.sum()
        centered_x, x_std = weighted_avg_and_std_torch(ground_truth, weights)
        centered_y, y_std = weighted_avg_and_std_torch(predictions, weights)
        covariance = (weights * centered_x * centered_y).sum(dim=dim, keepdim=True)
    bessel_corrected_covariance = covariance / (ground_truth.shape[dim] - 1)
    corr = bessel_corrected_covariance / (x_std * y_std)
    return corr


def CCC_torch(ground_truth, predictions, batch_first=False, weights=None):
    """
        Evaluates the Concordance Correlation Coefficient.
        Inputs are numpy arrays.
    """
    assert ground_truth.shape == predictions.shape
    assert predictions.numel() >= 2  # std doesn't make sense, unless there is at least two items in the batch
    if weights is not None:
        weights = weights / weights.sum()
        mean_pred, std_pred = weighted_avg_and_std_torch(predictions, weights)
        mean_gt, std_gt = weighted_avg_and_std_torch(ground_truth, weights)
    else:
        mean_pred = torch.mean(predictions)
        mean_gt = torch.mean(ground_truth)
        std_pred = torch.std(predictions)
        std_gt = torch.std(ground_truth)
    pearson = PCC_torch(ground_truth, predictions, batch_first=batch_first)
    return 2.0 * pearson * std_pred * std_gt / (std_pred ** 2 + std_gt ** 2 + (mean_pred - mean_gt) ** 2)


def weighted_avg_and_std_torch(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    weighted_mean = torch.sum(weights * values)
    weighted_std = torch.mean(weights * ((values - weighted_mean) ** 2))
    return weighted_mean, torch.sqrt(weighted_std)
