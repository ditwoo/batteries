# flake: noqa

import numpy as np

from batteries.metrics.utils import AverageMetter


def test_with_common_setup():
    metric = AverageMetter()

    assert metric.value == 0
    assert metric.counter == 0
    assert metric.sum == 0
    assert metric.average == 0

    values = np.random.randn(100)

    for v in values:
        metric.update(v)

    assert metric.value == values[-1]
    assert metric.counter == len(values)
    assert np.isclose(metric.sum, np.sum(values))
    assert np.isclose(metric.average, np.mean(values))

    metric.reset()

    assert metric.value == 0
    assert metric.counter == 0
    assert metric.sum == 0
    assert metric.average == 0


def test_with_inf():
    metric = AverageMetter()

    assert metric.value == 0
    assert metric.counter == 0
    assert metric.sum == 0
    assert metric.average == 0

    values = [1, 2, 3, float("inf")]

    for v in values:
        metric.update(v)

    assert metric.value == values[-1]
    assert metric.counter == len(values)
    assert np.isinf(metric.sum)
    assert np.isinf(metric.average)

    metric.reset()

    assert metric.value == 0
    assert metric.counter == 0
    assert metric.sum == 0
    assert metric.average == 0

    values = [1, 2, 3, -float("inf")]

    for v in values:
        metric.update(v)

    assert metric.value == values[-1]
    assert metric.counter == len(values)
    assert np.isinf(metric.sum)
    assert np.isinf(metric.average)

    metric.reset()

    assert metric.value == 0
    assert metric.counter == 0
    assert metric.sum == 0
    assert metric.average == 0


def test_with_nan():
    metric = AverageMetter()

    assert metric.value == 0
    assert metric.counter == 0
    assert metric.sum == 0
    assert metric.average == 0

    values = [1, 2, 3, float("nan")]

    for v in values:
        metric.update(v)

    assert np.isnan(metric.value)  # OR metric.value != metric.value
    assert metric.counter == len(values)
    assert np.isnan(metric.sum)
    assert np.isnan(metric.average)

    metric.reset()

    assert metric.value == 0
    assert metric.counter == 0
    assert metric.sum == 0
    assert metric.average == 0
