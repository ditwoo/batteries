# flake: noqa

from batteries.early_stop import EarlyStopIndicator


def test_minimization_stop_case():
    indicator = EarlyStopIndicator(patience=3, metric_minimization=True)

    values = [1.0, 1.0, 0.5, 0.7, 0.8, 0.9, 1.0, 1.0]
    expected = [False, False, False, False, False, False, True, True]
    actual = [indicator(num) for num in values]

    assert len(expected) == len(actual)
    assert expected == actual


def test_maximization_stop_case():
    indicator = EarlyStopIndicator(patience=2, metric_minimization=False)

    values = [0.4, 0.3, 0.2, 0.5, 0.2, 0.25, 0.1, 0.2]
    expected = [False, False, False, False, False, False, True, True]
    actual = [indicator(num) for num in values]

    assert len(expected) == len(actual)
    assert expected == actual
