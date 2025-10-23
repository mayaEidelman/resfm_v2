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
    pred_labels = (pred >= 0.6).float()

    # Base stats
    outliers_percent = (gt == 1).sum().item() / gt.shape[0]

    # Confusion matrix components
    tp = torch.sum((pred_labels == 1) & (gt == 1)).item()
    fp = torch.sum((pred_labels == 1) & (gt == 0)).item()
    tn = torch.sum((pred_labels == 0) & (gt == 0)).item()
    fn = torch.sum((pred_labels == 0) & (gt == 1)).item()
    # print('tp', tp, 'fp', fp, 'tn', tn, 'fn', fn)

    # Classification metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    recall_inliers = tn / (tn + fp + epsilon)             # Specificity
    outliers_after = fn / (tn + fn + epsilon)             # Outliers remaining after filtering

    # Score averages
    outliers_avg = pred[gt == 1].mean().item()            # Mean score for real outliers
    inliers_avg = pred[gt == 0].mean().item()             # Mean score for real inliers
    f_inliers_avg = pred[(pred_labels == 0) & (gt == 1)].mean().item()  # False inliers avg score
    # print('Accuracy', accuracy,
    #       'Precision', precision,
    #       'Recall', recall,
    #       'Recall(inliers)', recall_inliers,
    #       'F1_score', f1_score,
    #       '%outliers', outliers_percent,
    #       '%outliers_after', outliers_after,
    #       'outliers_avg', outliers_avg,
    #       'inliers_avg', inliers_avg,
    #       'f_inliers_avg', f_inliers_avg,)

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
    # gets list of dicts of the metrics
    # returns the dict of the means
    dict_mean_metrics = {}
    metrics = train_metrics[0].keys()
    for metric in metrics:
        # calculate the mean of each metric across all batches
        try:
            if phase is None:
                metric_mean = torch.mean(torch.tensor([e[metric] for e in train_metrics])).item()
                dict_mean_metrics[metric] = metric_mean
            else:
                if type(train_metrics[0][metric]) == type(1.0):
                    metric_mean = torch.mean(torch.tensor([e[metric] for e in train_metrics])).item()
                else:
                    metric_mean = torch.mean(torch.tensor([element for e in train_metrics for element in e[metric]])).item()

                dict_mean_metrics[phase.name + " - " + metric] = metric_mean
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error)
            print(train_metrics)



    return dict_mean_metrics