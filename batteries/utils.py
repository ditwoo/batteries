import os
import random
import shutil
from pathlib import Path
from collections import OrderedDict
from typing import Mapping, Any, Union, Sequence

from packaging.version import parse, Version
import numpy as np
import torch
from torch.backends import cudnn


def t2d(
    tensor: Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]],
    device: Union[str, torch.device],
) -> Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]]:
    """Move tensors to a specified device.

    Args:
        tensor (Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]]):
            data to move to a device.
        device (Union[str, torch.device]): device where should be moved device

    Returns:
        Union[torch.Tensor, Mapping[str, torch.Tensor], Sequence[torch.Tensor]]:
            data moved to a specified device
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


def make_checkpoint(
    stage, epoch, model, optimizer=None, scheduler=None, metrics=None
) -> dict:
    checkpoint = {
        "stage": stage,
        "epoch": epoch,
    }
    if isinstance(model, torch.nn.DataParallel):
        checkpoint["model_state_dict"] = model.module.state_dict()
    else:
        checkpoint["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if metrics is not None:
        checkpoint["metrics"] = metrics
    return checkpoint


def save_checkpoint(
    checkpoint: Mapping[str, Any],
    logdir: Union[str, Path],
    name: str,
    is_best: bool = False,
    is_last: bool = False,
) -> None:
    """Save checkpoint to a file.

    Args:
        checkpoint (Mapping[str, Any]): data to store in checkpoint
        logdir (Union[str, Path]): directory where should be stored checkpoint
        name (str): file name to use for storing checkpoint
        is_best (bool, optional): indicator to save checkpoint as best checkpoint.
            Defaults to False.
        is_last (bool, optional): indicator to save checkpoint as last checkpoint.
            Defaults to False.
    """
    os.makedirs(logdir, exist_ok=True)
    _name = name if name.endswith(".pth") else f"{name}.pth"
    filename = os.path.join(str(logdir), _name)
    torch.save(checkpoint, filename)
    if is_best:
        best_filename = os.path.join(str(logdir), "best.pth")
        shutil.copyfile(filename, best_filename)
    if is_last:
        last_filename = os.path.join(str(logdir), "last.pth")
        shutil.copyfile(filename, last_filename)


def checkpoints_weight_average(*files) -> OrderedDict:
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
        files: an iterable of string paths of checkpoints to load from.

    Returns:
        A dict of string keys mapping to various values. The 'model' key
        from the returned dict should correspond to an OrderedDict mapping
        string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(files)

    for f in files:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state["model_state_dict"]

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k].div_(num_models)
    new_state["model_state_dict"] = averaged_params
    return new_state


def seed_all(seed: int = 42) -> None:
    """Fix all seeds so results can be reproducible.

    Args:
        seed (int): random seed
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

