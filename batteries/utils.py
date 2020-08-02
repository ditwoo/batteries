import os
import torch
import shutil
from pathlib import Path
from typing import Mapping, Any, Union, Sequence


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
