import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from batteries import (
    AverageMetter,
    CheckpointManager,
    TensorboardLogger,
    load_checkpoint,
    make_checkpoint,
    seed_all,
    t2d,
)
from batteries.distributed import all_gather, mreduce
from batteries.progress import tqdm

from datasets import get_loaders
from models import SimpleNet


torch.backends.cudnn.benchmark = True


def setup(rank, world_size):
    """Initialize distributed experiment.

    Args:
        rank (int): process rank
        world_size (int): total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12346"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Close distributed experiment."""
    dist.destroy_process_group()


def train_fn(
    model,
    loader,
    world_setup,
    loss_fn,
    optimizer,
    scheduler=None,
    accum_steps=1,
    tb_logger=None,
    last_iteration_index=None,
):
    """One training epoch.

    Args:
        model (torcn.nn.Module): model to train
        loader (torch.utils.data.Dataloader): loader to use for training
        world_setup (tuple): (rank, world size)
        loss_fn (torch.nn.Module or callable): loss function
        optimizer (torch.optim.Optimizer): model optimizer
        scheduler (): batch scheduler.
            Default is `None`.
        accum_steps (int): number of steps to accumulate gradients.
            Default is `1`.
        tb_logger (batteries.TensorboardLogger or tensorboardx.SummaryWriter):
            writer for storing batch values.
            Default is `None`.
        last_iteration_index (int): index of last iteration (used with `tb_logger`).
            Default is `None`.

    Returns:
        dict with training metrics (key - str, value - float)
    """
    model.train()

    local_rank, world_size = world_setup
    verbose = local_rank == 0
    last_iteration_index = last_iteration_index or 0

    num_batches = len(loader)

    metrics = {"loss": AverageMetter(), "predicted": [], "true": []}

    for _idx, (inputs, targets) in enumerate(loader):
        inputs, targets = t2d((inputs, targets), "cuda")  # move to default CUDA device

        if isinstance(inputs, (tuple, list)):
            outputs = model(*inputs)
        elif isinstance(inputs, dict):
            outputs = model(**inputs)
        else:
            outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        metrics["predicted"].append(targets.flatten().detach().cpu().numpy())
        metrics["true"].append(outputs.argmax(1).flatten().detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()

        if (_idx + 1) % accum_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        torch.cuda.synchronize()

        reduced_loss = mreduce(loss, world_size)
        metrics["loss"].update(reduced_loss.item(), inputs.size(0))

        if verbose:
            print("loss {:.4f}".format(metrics["loss"].average), end="\r")

            if tb_logger is not None:
                tb_logger.add_scalar(
                    "batch/train/loss",
                    metrics["loss"].average,
                    last_iteration_index + _idx + 1,
                )

    metrics["true"] = np.concatenate(metrics["true"])
    metrics["predicted"] = np.concatenate(metrics["predicted"])

    true_labels = np.concatenate(all_gather(metrics["true"]))
    predicted_labels = np.concatenate(all_gather(metrics["predicted"]))

    dataset_acc = (true_labels == predicted_labels).astype(float).mean()

    return {"loss": metrics["loss"].average, "accuracy": dataset_acc}


@torch.no_grad()
def valid_fn(
    model, loader, world_setup, loss_fn, tb_logger=None, last_iteration_index=None
):
    """One validation epoch.

    Args:
        model (torcn.nn.Module): model to train
        loader (torch.utils.data.Dataloader): loader to use for training
        world_setup (tuple): (rank, world size)
        loss_fn (torch.nn.Module or callable): loss function
        tb_logger (batteries.TensorboardLogger or tensorboardx.SummaryWriter):
            writer for storing batch values.
            Default is `None`.
        last_iteration_index (int): index of last iteration (used with `tb_logger`).
            Default is `None`.

    Returns:
        dict with validation metrics (key - str, value - float)
    """
    model.eval()

    local_rank, world_size = world_setup
    last_iteration_index = last_iteration_index or 0

    num_batches = len(loader)
    verbose = local_rank == 0

    metrics = {"loss": AverageMetter(), "predicted": [], "true": []}

    for _idx, (inputs, targets) in enumerate(loader):
        inputs, targets = t2d((inputs, targets), "cuda")

        if isinstance(inputs, (tuple, list)):
            outputs = model(*inputs)
        elif isinstance(inputs, dict):
            outputs = model(**inputs)
        else:
            outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        metrics["predicted"].append(targets.flatten().detach().cpu().numpy())
        metrics["true"].append(outputs.argmax(1).flatten().detach().cpu().numpy())

        torch.cuda.synchronize()

        reduced_loss = mreduce(loss, world_size)
        metrics["loss"].update(reduced_loss.item(), inputs.size(0))

        if verbose:
            print("loss {:.4f}".format(metrics["loss"].average), end="\r")

            if tb_logger is not None:
                tb_logger.add_metric(
                    "batch/valid/loss",
                    metrics["loss"].average,
                    last_iteration_index + _idx + 1,
                )

    metrics["true"] = np.concatenate(metrics["true"])
    metrics["predicted"] = np.concatenate(metrics["predicted"])

    true_labels = np.concatenate(all_gather(metrics["true"]))
    predicted_labels = np.concatenate(all_gather(metrics["predicted"]))

    dataset_acc = (true_labels == predicted_labels).astype(float).mean()

    return {"loss": metrics["loss"].average, "accuracy": dataset_acc}


def experiment(rank, world_size, logdir):
    """Experiment flow.

    Args:
        rank (int): process rank
        world_size (int): world size
        logdir (pathlib.Path): directory with logs
    """
    # preparations
    torch.cuda.set_device(rank)
    setup(rank, world_size)
    logdir = Path(logdir) if isinstance(logdir, str) else logdir
    tb_logdir = logdir / "tensorboard"

    main_metric = "accuracy"
    minimize_metric = False

    def log(text):
        if rank == 0:
            print(text)

    train_loader, valid_loader = get_loaders("", rank, world_size)
    world_setup = (rank, world_size)

    train_batch_cnt = 0
    valid_batch_cnt = 0

    with TensorboardLogger(str(tb_logdir), write_to_disk=(rank == 0)) as tb:
        stage = "stage0"
        n_epochs = 2
        log(f"Stage - {stage}")

        seed_all()
        model = SimpleNet()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        log("Used sync batchnorm")

        model = model.to(rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        checkpointer = CheckpointManager(
            logdir=logdir / stage,
            metric=main_metric,
            metric_minimization=minimize_metric,
            save_n_best=3,
        )

        for ep in range(1, n_epochs + 1):
            log(f"[Epoch {ep}/{n_epochs}]")
            train_metrics = train_fn(
                model,
                train_loader,
                world_setup,
                criterion,
                optimizer,
                tb_logger=tb,
                last_iteration_index=train_batch_cnt,
            )
            if rank == 0:
                tb.add_scalars(f"{stage}/train", train_metrics, ep)
            train_batch_cnt += len(train_loader)

            valid_metrics = valid_fn(
                model,
                valid_loader,
                world_setup,
                criterion,
                tb_logger=tb,
                last_iteration_index=valid_batch_cnt,
            )
            valid_batch_cnt += len(valid_loader)
            if rank == 0:
                tb.add_scalars(f"{stage}/valid", valid_metrics, ep)

                # store checkpoints
                checkpointer.process(
                    score=valid_metrics[main_metric],
                    epoch=ep,
                    checkpoint=make_checkpoint(
                        stage,
                        ep,
                        model,
                        optimizer,
                        metrics={"train": train_metrics, "valid": valid_metrics},
                    ),
                )

            log(
                "[{}/{}] train: loss - {}, accuracy - {}".format(
                    ep, n_epochs, train_metrics["loss"], train_metrics["accuracy"]
                )
            )
            log(
                "[{}/{}] valid: loss - {}, accuracy - {}".format(
                    ep, n_epochs, valid_metrics["loss"], valid_metrics["accuracy"]
                )
            )

        # do a next training stage
        stage = "stage1"
        n_epochs = 3
        log("*" * 100)
        log(f"Stage - {stage}")

        # wait other processes
        dist.barrier()

        model = SimpleNet()
        load_checkpoint(logdir / "stage0" / "best.pth", model, verbose=True)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = model.to(rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        optimizer = optim.Adam(model.parameters(), lr=1e-4 / 2)

        checkpointer = CheckpointManager(
            logdir=logdir / stage,
            metric=main_metric,
            metric_minimization=minimize_metric,
            save_n_best=3,
        )

        for ep in range(1, n_epochs + 1):
            log(f"[Epoch {ep}/{n_epochs}]")
            train_metrics = train_fn(
                model,
                train_loader,
                world_setup,
                criterion,
                optimizer,
                tb_logger=tb,
                last_iteration_index=train_batch_cnt,
            )
            if rank == 0:
                tb.add_scalars(f"{stage}/train", train_metrics, ep)
            train_batch_cnt += len(train_loader)

            valid_metrics = valid_fn(
                model,
                valid_loader,
                world_setup,
                criterion,
                tb_logger=tb,
                last_iteration_index=valid_batch_cnt,
            )
            valid_batch_cnt += len(valid_loader)
            if rank == 0:
                tb.add_scalars(f"{stage}/valid", valid_metrics, ep)

                # store checkpoints
                checkpointer.process(
                    score=valid_metrics[main_metric],
                    epoch=ep,
                    checkpoint=make_checkpoint(
                        stage,
                        ep,
                        model,
                        optimizer,
                        metrics={"train": train_metrics, "valid": valid_metrics},
                    ),
                )

            log(
                "[{}/{}] train: loss - {}, accuracy - {}".format(
                    ep, n_epochs, train_metrics["loss"], train_metrics["accuracy"]
                )
            )
            log(
                "[{}/{}] valid: loss - {}, accuracy - {}".format(
                    ep, n_epochs, valid_metrics["loss"], valid_metrics["accuracy"]
                )
            )

    cleanup()


def main() -> None:
    logdir = Path(".") / "logs" / "ddp-experiment"
    world_size = torch.cuda.device_count()

    if os.path.isdir(logdir):
        shutil.rmtree(logdir, ignore_errors=True)
        print(f"* Removed existing '{logdir}' directory!")

    os.makedirs(str(logdir))

    mp.spawn(experiment, args=(world_size, logdir,), nprocs=world_size, join=True)

    # do some post cleaning


if __name__ == "__main__":
    main()
