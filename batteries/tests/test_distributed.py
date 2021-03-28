# flake: noqa

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from batteries.distributed import all_gather, mreduce, sreduce

if torch.cuda.is_available():
    IS_MULTIPLE_CUDA_DEVICES = torch.cuda.device_count() > 1
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
else:
    IS_MULTIPLE_CUDA_DEVICES = False


def _setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _cleanup():
    dist.destroy_process_group()


def _mreduce(rank, world_size):
    _setup(rank, world_size)

    to_mreduce = torch.tensor(rank + 1).float().to(rank)
    actual = mreduce(to_mreduce, world_size)

    assert actual == torch.tensor(((world_size + 1) / 2), dtype=torch.float).to(rank)

    _cleanup()


def _sreduce(rank, world_size):
    _setup(rank, world_size)

    to_sreduce = torch.tensor(rank + 1, dtype=torch.int).to(rank)
    actual = sreduce(to_sreduce)

    assert actual == torch.tensor((world_size * (world_size + 1)) // 2, dtype=torch.int).to(rank)

    _cleanup()


def _all_gather(rank, world_size):
    _setup(rank, world_size)

    to_gather = torch.ones(3, dtype=torch.int) * (rank + 1)  # use cpu tensors
    actual = all_gather(to_gather)
    actual = torch.cat(actual)

    expected = torch.cat([torch.ones(3, dtype=torch.int) * (i + 1) for i in range(world_size)])

    assert torch.all(actual.eq(expected))

    _cleanup()


def _run_test(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


@pytest.mark.skipif(not IS_MULTIPLE_CUDA_DEVICES, reason="need at least 2 cuda devices")
def test_mreduce():
    n_gpus = torch.cuda.device_count()
    _run_test(_mreduce, n_gpus)


@pytest.mark.skipif(not IS_MULTIPLE_CUDA_DEVICES, reason="need at least 2 cuda devices")
def test_sreduce():
    n_gpus = torch.cuda.device_count()
    _run_test(_sreduce, n_gpus)


@pytest.mark.skipif(not IS_MULTIPLE_CUDA_DEVICES, reason="need at least 2 cuda devices")
def test_all_gather():
    n_gpus = torch.cuda.device_count()
    _run_test(_all_gather, n_gpus)
