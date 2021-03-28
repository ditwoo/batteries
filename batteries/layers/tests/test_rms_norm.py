# flake: noqa

import numpy as np
import torch

from batteries.layers.rms_norm import RMSNorm


def test_rms_norm_values():
    in_features = 13
    eps = 1e-6

    layer = RMSNorm(in_features, bias=False, eps=eps).eval()

    x = np.random.randn(3, 5, in_features)

    expected = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    expected = expected * in_features ** (-0.5)
    expected = x / (expected + eps)

    actual = layer(torch.from_numpy(x)).detach().cpu().numpy()
    assert np.allclose(expected, actual)

    layer = RMSNorm(in_features, bias=True, eps=eps).eval()

    x = np.random.randn(3, 5, in_features)

    expected = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    expected = expected * in_features ** (-0.5)
    expected = x / (expected + eps)

    actual = layer(torch.from_numpy(x)).detach().cpu().numpy()
    assert np.allclose(expected, actual)


def test_rms_norm_partial_values():
    in_features = 20
    partial = 0.4
    partial_features = int(in_features * partial)
    eps = 1e-7

    layer = RMSNorm(in_features, partial=partial, bias=False, eps=eps).eval()

    x = np.random.randn(7, 11, in_features)
    partial_x = x[:, :, :partial_features]

    expected = np.linalg.norm(partial_x, ord=2, axis=-1, keepdims=True)
    expected = expected * partial_features ** (-0.5)
    expected = x / (expected + eps)

    actual = layer(torch.from_numpy(x)).detach().cpu().numpy()

    assert np.allclose(expected, actual)

    layer = RMSNorm(in_features, partial=partial, bias=True, eps=eps).eval()

    x = np.random.randn(7, 11, in_features)
    partial_x = x[:, :, :partial_features]

    expected = np.linalg.norm(partial_x, ord=2, axis=-1, keepdims=True)
    expected = expected * partial_features ** (-0.5)
    expected = x / (expected + eps)

    actual = layer(torch.from_numpy(x)).detach().cpu().numpy()

    assert np.allclose(expected, actual)
