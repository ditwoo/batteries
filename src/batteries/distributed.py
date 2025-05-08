import pickle
from contextlib import contextmanager

import torch
from torch import distributed as dist

__all__ = ("sreduce", "mreduce", "all_gather", "zero_rank_first")


def sreduce(tensor):
    """Sum reduce.

    Args:
        tensor (torch.Tensor): data to reduce
        num (int): number of devices

    Returns:
        reduced tensor value
    """
    _clone = tensor.clone()
    dist.all_reduce(_clone, dist.ReduceOp.SUM)
    return _clone


def mreduce(tensor, num):
    """Mean reduce.

    Args:
        tensor (torch.Tensor): data to reduce
        num (int): number of devices

    Returns:
        reduced tensor value
    """
    _clone = tensor.clone()
    dist.all_reduce(_clone, dist.ReduceOp.SUM)
    _clone /= num
    return _clone


def all_gather(data):
    """Run all_gather on arbitrary picklable data (not necessarily tensors).

    NOTE: if data on different devices then data in resulted list will
        be on the same devices.

    Source: https://github.com/facebookresearch/detr/blob/master/util/misc.py#L88-L128

    Args:
        data: any picklable object

    Returns:
        list[data]: list of data gathered from each rank
    """
    if not dist.is_available() or not dist.is_initialized():
        world_size = 1
    else:
        world_size = dist.get_world_size()

    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


@contextmanager
def zero_rank_first(local_rank):
    """Decorator which makes sure that process with local_rank == 0
    will do some code block first and wait for other processes to finish.

    Example:
        >>> # somewhere in DDP code
        >>> import torchvision
        >>> # master process will load data and other processes will use cache
        >>> with zero_rank_first(local_rank):
        >>>     train_dataset = torchvision.datasets.CIFAR10("/cifat10", train=True, download=True)
        >>>     valid_dataset = torchvision.datasets.CIFAR10("/cifat10", train=False, download=True)

    Args:
        local_rank (int): process rank.
    """
    if local_rank not in {-1, 0}:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()
