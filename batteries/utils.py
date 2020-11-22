import random
from typing import Mapping, Sequence, Union

import numpy as np
import torch
from packaging.version import Version, parse
from torch.backends import cudnn


def t2d(tensor, device):
    """Move tensors to a specified device.

    Args:
        tensor (torch.Tensor or Dict[str, torch.Tensor] or list/tuple of torch.Tensor):
            data to move to a device.
        device (str or torch.device): device where should be moved device

    Returns:
        torch.Tensor or Dict[str, torch.Tensor] or list/tuple of torch.Tensor
        based on `tensor` type with data moved to a specified device
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, (tuple, list)):
        # recursive move to device
        return [t2d(_tensor, device) for _tensor in tensor]
    elif isinstance(tensor, dict):
        res = {}
        for _key, _tensor in tensor.items():
            res[_key] = t2d(_tensor, device)
        return res


def seed_all(seed=42) -> None:
    """Fix all seeds so results can be reproducible.

    Args:
        seed (int): random seed.
            Default is 42.
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        if parse(tf.__version__) >= Version("2.0.0"):
            tf.random.set_seed(seed)
        elif parse(tf.__version__) <= Version("1.13.2"):
            tf.set_random_seed(seed)
        else:
            tf.compat.v1.set_random_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    # reproducibility
    cudnn.deterministic = True
    # small speedup
    cudnn.benchmark = False


def zero_grad(optimizer) -> None:
    """Perform an hacky way to zero gradients.

    Args:
        optimizer (torch.optim.Optimizer): optimizer with model parameters.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = None
