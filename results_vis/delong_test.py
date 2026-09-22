"""
DeLong's test for comparing two correlated ROC curves.

Based on:
DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more
correlated receiver operating characteristic curves: a nonparametric approach.
Biometrics. 1988;44(3):837-845.

Implementation adapted from scipy and various scientific Python implementations.
"""

import numpy as np
from scipy import stats


def compute_midrank(x):
    """Compute midrank of values in x."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.

    Args:
        predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
            sorted such that the examples with label "1" are first
    Returns:
        (AUC value, AUC covariance)
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values."""
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes DeLong test for the comparison of two correlated ROC curves.

    Args:
        ground_truth: 1D array of binary ground truth labels (0 or 1)
        predictions_one: 1D array of predictions from model 1
        predictions_two: 1D array of predictions from model 2

    Returns:
        log10_pvalue: log10 of the p-value
        auc_diff: AUC difference (model 1 - model 2)
        z_score: z-score from the test
        pvalue: actual p-value
    """
    ground_truth = np.array(ground_truth)
    predictions_one = np.array(predictions_one)
    predictions_two = np.array(predictions_two)

    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)

    # Compute z-score and p-value
    l = np.array([[1, -1]])
    z_score = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, delongcov), l.T))
    z_score = float(z_score.squeeze())
    pvalue = 2 * (1 - stats.norm.cdf(z_score))  # two-tailed
    log10_pvalue = np.log10(pvalue) if pvalue > 0 else -np.inf
    auc_diff = float(aucs[0] - aucs[1])

    return {
        'log10_pvalue': float(log10_pvalue),
        'pvalue': float(pvalue),
        'z_score': float(z_score),
        'auc_diff': auc_diff,
        'auc1': float(aucs[0]),
        'auc2': float(aucs[1])
    }


if __name__ == '__main__':
    # Test example
    np.random.seed(42)
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_pred1 = np.random.rand(n)
    y_pred2 = np.random.rand(n)

    result = delong_roc_test(y_true, y_pred1, y_pred2)
    print("DeLong test results:")
    print(f"  AUC 1: {result['auc1']:.4f}")
    print(f"  AUC 2: {result['auc2']:.4f}")
    print(f"  AUC diff: {result['auc_diff']:.4f}")
    print(f"  z-score: {result['z_score']:.4f}")
    print(f"  p-value: {result['pvalue']:.4e}")
    print(f"  log10(p): {result['log10_pvalue']:.4f}")
