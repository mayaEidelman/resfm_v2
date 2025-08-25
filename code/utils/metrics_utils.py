import torch
from utils.Phases import Phases

def nanMetrics():
    dict = {'Accuracy': 0.0, 'Precision':0.0, 'Recall':0.0,
    'Recall(inliers)':0.0, 'F1_score':0.0, '%outliers':0.0,
    '%outliers_after':0.0, 'outliers_avg':0.0, 'inliers_avg':0.0,
    'f_inliers_avg':0.0}

    return dict

def classificationMetrics(pred, gt):
    """
    Compute classification metrics for binary outlier prediction.

    Positive label = outlier (1)
    Negative label = inlier (0)

    Args:
        pred (Tensor): Predicted scores (float), one per valid observation.
        gt (Tensor): Ground truth binary labels (1 = outlier, 0 = inlier).

    Returns:
        dict: A dictionary containing accuracy, precision, recall, F1, and error statistics.
    """
    epsilon = 1e-20

    # Convert predicted scores to binary labels
    pred_labels = (pred >= 0.5).float()

    # Base stats
    outliers_percent = (gt == 1).sum().item() / gt.shape[0]

    # Confusion matrix components
    tp = torch.sum((pred_labels == 1) & (gt == 1)).item()
    fp = torch.sum((pred_labels == 1) & (gt == 0)).item()
    tn = torch.sum((pred_labels == 0) & (gt == 0)).item()
    fn = torch.sum((pred_labels == 0) & (gt == 1)).item()

    # Classification metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    recall_inliers = tn / (tn + fp + epsilon)             # Specificity
    outliers_after = fn / (tn + fn + epsilon)             # Outliers remaining after filtering

    # Score averages with empty-mask guards (avoid NaNs)
    out_mask = (gt == 1)
    in_mask = (gt == 0)
    false_inlier_mask = (pred_labels == 0) & (gt == 1)

    outliers_avg = pred[out_mask].mean().item() if out_mask.any() else 0.0
    inliers_avg = pred[in_mask].mean().item() if in_mask.any() else 0.0
    f_inliers_avg = pred[false_inlier_mask].mean().item() if false_inlier_mask.any() else 0.0

        # Score averages
    # outliers_avg = pred[gt == 1].mean().item()            # Mean score for real outliers
    # inliers_avg = pred[gt == 0].mean().item()             # Mean score for real inliers
    # f_inliers_avg = pred[(pred_labels == 0) & (gt == 1)].mean().item()  # False inliers avg score

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Recall(inliers)': recall_inliers,
        'F1_score': f1_score,
        '%outliers': outliers_percent,
        '%outliers_after': outliers_after,
        'outliers_avg': outliers_avg,
        'inliers_avg': inliers_avg,
        'f_inliers_avg': f_inliers_avg,
    }

def OutliersMetrics(pred_outliers, data):
    """
    Compute classification metrics for predicted outliers in a scene.

    Args:
        pred_outliers (Tensor): Predicted outlier scores for valid 2D points (shape: [#valid entries]).
        data: SceneData object containing ground truth outlier matrix (data.outlier_indices)
              and the (i,j) indices of valid 2D points (data.x.indices).

    Returns:
        dict: A dictionary of classification metrics (accuracy, precision, recall, etc.).
    """
    # Get ground truth labels for only the valid entries (according to sparse indexing)
    img_ids, pt_ids = data.x.indices.T[:, 0], data.x.indices.T[:, 1]
    gt_outliers = data.outlier_indices[img_ids, pt_ids]

    # Compute classification metrics
    metrics = classificationMetrics(pred_outliers.squeeze(), gt_outliers.float())

    return metrics

def CalcMeanBatchMetrics(train_metrics, phase=None):
    # Accepts either a list[dict], or list[list[dict]] (e.g., gathered across ranks)
    # Returns a dict of mean metrics
    dict_mean_metrics = {}

    # Flatten one level if needed
    if isinstance(train_metrics, list) and len(train_metrics) > 0 and isinstance(train_metrics[0], list):
        flat_metrics = [d for sub in train_metrics for d in sub]
    else:
        flat_metrics = train_metrics

    if not flat_metrics:
        return dict_mean_metrics

    # Determine metrics keys from the first dict
    metrics_keys = list(flat_metrics[0].keys())

    for metric in metrics_keys:
        try:
            values = []
            for entry in flat_metrics:
                v = entry[metric]
                # Normalize to Python float
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        v = v.item()
                    else:
                        # If somehow an array sneaks in, take its mean
                        v = v.float().mean().item()
                values.append(float(v))

            mean_val = float(torch.tensor(values).mean().item())
            key = metric if phase is None else (phase.name + " - " + metric)
            dict_mean_metrics[key] = mean_val
        except Exception as error:
            print("An exception occurred:", error)
            print(flat_metrics)

    return dict_mean_metrics