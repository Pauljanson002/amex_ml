import numpy as np


def amex_metric(y_true, y_pred):
    labels = np.array([y_true, y_pred]).T
    labels = labels[np.argsort(-y_pred)]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])
    gini = [0, 0]
    for i in [1, 0]:
        labels = np.array([y_true, y_pred]).T
        labels = labels[np.argsort(-y_pred if i else -y_true)]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1] / gini[0] + top_four)


def lightgbm_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    metric_value = amex_metric(y_true, y_pred)
    return "amex_metric", metric_value, True
