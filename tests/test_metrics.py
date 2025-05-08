# flake8: noqa

import numpy as np
import torch
from sklearn import metrics

from batteries.metrics.torch import binary_fbeta, binary_precision, binary_recall, classification_accuracy


def _fix_seed():
    np.random.seed(2020)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_accuracy():
    _fix_seed()
    size = 1000

    preds = np.random.randint(0, 1000, size=size)
    ground_truth = np.random.randint(0, 1000, size=size)

    expected = metrics.accuracy_score(ground_truth, preds)
    actual = classification_accuracy(y_true=torch.from_numpy(ground_truth), y_pred=torch.from_numpy(preds))

    assert np.allclose(expected, actual)


def test_precision():
    _fix_seed()
    size = 1000
    threshold = 0.5
    logits = np.random.randn(size)
    preds = (_sigmoid(logits) >= threshold).astype(int)
    ground_truth = np.random.randint(0, 2, size=size)

    expected = metrics.precision_score(ground_truth, preds)
    actual = binary_precision(
        y_pred=torch.from_numpy(logits),
        y_true=torch.from_numpy(ground_truth),
        threshold=0.5,
    )

    assert np.allclose(expected, actual)


def test_recall():
    _fix_seed()
    size = 1000
    threshold = 0.5
    logits = np.random.randn(size)
    preds = (_sigmoid(logits) >= threshold).astype(int)
    ground_truth = np.random.randint(0, 2, size=size)

    expected = metrics.recall_score(ground_truth, preds)
    actual = binary_recall(
        y_pred=torch.from_numpy(logits),
        y_true=torch.from_numpy(ground_truth),
        threshold=0.5,
    )

    assert np.allclose(expected, actual)


def test_f1():
    _fix_seed()
    size = 1000
    threshold = 0.5
    logits = np.random.randn(size)
    preds = (_sigmoid(logits) >= threshold).astype(int)
    ground_truth = np.random.randint(0, 2, size=size)

    expected = metrics.f1_score(ground_truth, preds)
    actual = binary_fbeta(
        y_pred=torch.from_numpy(logits),
        y_true=torch.from_numpy(ground_truth),
        threshold=0.5,
        beta=1.0,
    )

    assert np.allclose(expected, actual)


def test_f2():
    _fix_seed()
    size = 1000
    threshold = 0.5
    logits = np.random.randn(size)
    preds = (_sigmoid(logits) >= threshold).astype(int)
    ground_truth = np.random.randint(0, 2, size=size)

    expected = metrics.fbeta_score(ground_truth, preds, beta=2.0)
    actual = binary_fbeta(
        y_pred=torch.from_numpy(logits),
        y_true=torch.from_numpy(ground_truth),
        threshold=0.5,
        beta=2.0,
    )

    assert expected == actual
