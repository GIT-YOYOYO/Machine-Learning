"""分类模型的评估指标，接口与 sklearn.metrics 对齐。"""

import numpy as np


def confusion_matrix(y_true, y_pred, labels=None):
    """返回混淆矩阵，行是真实类别，列是预测类别。

    对角线是预测正确的样本数，非对角线是错分样本数。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[index[true_label], index[pred_label]] += 1
    return matrix


def accuracy_score(y_true, y_pred):
    """预测正确的样本比例。"""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))