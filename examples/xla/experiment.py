import os
import sys
import shutil
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from datasets import get_loaders
from models import SimpleNet

from batteries import (
    seed_all,
    CheckpointManager,
    TensorboardLogger,
    t2d,
    make_checkpoint,
    load_checkpoint,
)
from batteries.progress import tqdm


######################################################################
# TODOs:
# 1. save tensorboard metrics after each step (train/valid)
# 2. typings and docs to each function
######################################################################


def train_fn(
    model,
    loader,
    device,
    loss_fn,
    optimizer,
    scheduler=None,
    accum_steps: int = 1,
    verbose: bool = True,
):
    model.train()

    verbose_on_device = verbose and device == 0

    losses = []
    dataset_true_lbl = []
    dataset_pred_lbl = []

    with tqdm(
        desc="train", disable=not verbose_on_device
    ) as progress:
        for _idx, (bx, by) in enumerate(loader.per_device_loader(device)):
            # bx, by = t2d((bx, by), device)

            optimizer.zero_grad()

            if isinstance(bx, (tuple, list)):
                outputs = model(*bx)
            elif isinstance(bx, dict):
                outputs = model(**bx)
            else:
                outputs = model(bx)

            loss = loss_fn(outputs, by)
            _loss = loss.detach().item()
            losses.append(_loss)
            loss.backward()

            dataset_true_lbl.append(by.flatten().detach().cpu().numpy())
            dataset_pred_lbl.append(outputs.argmax(1).flatten().detach().cpu().numpy())

            if verbose_on_device:
                progress.set_postfix_str(f"loss {_loss:.4f}")

            if (_idx + 1) % accum_steps == 0:
                xm.optimizer_step(optimizer)
                if scheduler is not None:
                    scheduler.step()

            progress.update(1)

    dataset_true_lbl = np.concatenate(dataset_true_lbl)
    dataset_pred_lbl = np.concatenate(dataset_pred_lbl)
    dataset_acc = (dataset_true_lbl == dataset_pred_lbl).astype(float).mean()

    return np.mean(losses), dataset_acc


def valid_fn(model, loader, device, loss_fn, verbose: bool = True):
    model.eval()

    verbose_on_device = verbose and device == 0

    losses = []
    dataset_true_lbl = []
    dataset_pred_lbl = []
    with torch.no_grad(), tqdm(
        desc="valid", disable=not verbose_on_device
    ) as progress:
        to_iter = loader
        for bx, by in loader.per_device_loader(device):
            # bx, by = t2d((bx, by), device)

            if isinstance(bx, (tuple, list)):
                outputs = model(*bx)
            elif isinstance(bx, dict):
                outputs = model(**bx)
            else:
                outputs = model(bx)

            loss = loss_fn(outputs, by).detach().item()
            losses.append(loss)

            if verbose_on_device:
                progress.set_postfix_str(f"loss {loss:.4f}")

            dataset_true_lbl.append(by.flatten().detach().cpu().numpy())
            dataset_pred_lbl.append(outputs.argmax(1).flatten().detach().cpu().numpy())

            progress.update(1)

    dataset_true_lbl = np.concatenate(dataset_true_lbl)
    dataset_pred_lbl = np.concatenate(dataset_pred_lbl)
    dataset_acc = (dataset_true_lbl == dataset_pred_lbl).astype(float).mean()

    return np.mean(losses), dataset_acc


def experiment(rank: int, logdir: str) -> None:
    device = xm.xla_device()
    tb_logdir = logdir / "tensorboard"

    def pprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    seed_all()
    model = SimpleNet()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader, valid_loader = get_loaders("")

    with TensorboardLogger(tb_logdir) as tb:
        stage = "stage0"
        n_epochs = 10

        checkpointer = CheckpointManager(
            logdir=logdir / stage,
            metric="accuracy",
            metric_minimization=False,
            save_n_best=3,
            save_fn=xm.save
        )

        for ep in range(1, n_epochs + 1):
            
            pl_train_loader = pl.ParallelLoader(train_loader, [device])
            pl_valid_loader = pl.ParallelLoader(valid_loader, [device])

            pprint(f"[Epoch {ep}/{n_epochs}]")
            train_loss, train_acc = train_fn(
                model, pl_train_loader, device, criterion, optimizer
            )
            valid_loss, valid_acc = valid_fn(model, pl_valid_loader, device, criterion)

            if rank == 0:
                # log metrics
                tb.metric(
                    f"{stage}/loss", {"train": train_loss, "valid": valid_loss}, ep
                )
                tb.metric(
                    f"{stage}/accuracy", {"train": train_acc, "valid": valid_acc}, ep,
                )

                epoch_metrics = {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_acc,
                }

                # # store checkpoints
                # checkpointer.process(
                #     metric_value=valid_acc,
                #     epoch=ep,
                #     checkpoint=make_checkpoint(
                #         stage, ep, model, optimizer, metrics=epoch_metrics,
                #     ),
                # )

            pprint(f"            train loss - {train_loss:.5f}")
            pprint(f"train dataset accuracy - {train_acc:.5f}")
            pprint(f"            valid loss - {valid_loss:.5f}")
            pprint(f"valid dataset accuracy - {valid_acc:.5f}")
            pprint()

        # # do a next training stage
        # stage = "stage1"
        # n_epochs = 10
        # pprint(f"\n\nStage - {stage}")

        # dist.barrier()
        # model = SimpleNet()
        # load_checkpoint(logdir / "stage0" / "best.pth", model)
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # model = model.to(rank)
        # model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        # optimizer = optim.Adam(model.parameters(), lr=1e-4 / 2)

        # # checkpointer = CheckpointManager(
        # #     logdir=logdir / stage,
        # #     metric="accuracy",
        # #     metric_minimization=False,
        # #     save_n_best=3,
        # # )

        # for ep in range(1, n_epochs + 1):
        #     pprint(f"[Epoch {ep}/{n_epochs}]")
        #     train_loss, train_acc = train_fn(
        #         model, train_loader, rank, criterion, optimizer
        #     )
        #     valid_loss, valid_acc = valid_fn(model, valid_loader, rank, criterion)

        #     if rank == 0:
        #         # log metrics
        #         tb.metric(
        #             f"{stage}/loss", {"train": train_loss, "valid": valid_loss}, ep
        #         )
        #         tb.metric(
        #             f"{stage}/accuracy", {"train": train_acc, "valid": valid_acc}, ep,
        #         )

        #         epoch_metrics = {
        #             "train_loss": train_loss,
        #             "train_accuracy": train_acc,
        #             "valid_loss": valid_loss,
        #             "valid_accuracy": valid_acc,
        #         }

        #         # # store checkpoints
        #         # checkpointer.process(
        #         #     metric_value=valid_acc,
        #         #     epoch=ep,
        #         #     checkpoint=make_checkpoint(
        #         #         stage, ep, model, optimizer, metrics=epoch_metrics,
        #         #     ),
        #         # )

        #     pprint(f"            train loss - {train_loss:.5f}")
        #     pprint(f"train dataset accuracy - {train_acc:.5f}")
        #     pprint(f"            valid loss - {valid_loss:.5f}")
        #     pprint(f"valid dataset accuracy - {valid_acc:.5f}")
        #     pprint()


def main() -> None:
    # do some pre cleaning & stuff
    logdir = Path(".") / "logs" / "xla-experiment"

    if os.path.isdir(logdir):
        shutil.rmtree(logdir, ignore_errors=True)
        print(f"* Removed existing '{logdir}' directory!")

    xmp.spawn(experiment, args=(logdir,))

    # do some post cleaning


if __name__ == "__main__":
    main()
