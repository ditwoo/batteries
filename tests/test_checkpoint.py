import os
import numpy as np
from tempfile import TemporaryDirectory

import torch
from batteries.checkpoints import CheckpointManager


def test_checkpoint_manager_metric_maximization():
    n_best = np.random.randint(5, 20)
    minimize = False
    metric_name = "accuracy"
    file_prefix = "experiment"
    metrics = np.random.uniform(size=100)

    best_metric_epochs = np.argsort(metrics)
    if not minimize:
        best_metric_epochs = best_metric_epochs[::-1]
    best_metric_epochs = best_metric_epochs[:n_best]
    best_metrics = metrics[best_metric_epochs]
    best_metric_epochs += 1

    expected_files = ["metrics.json", "last.pth", "best.pth"] + [
        f"experiment_{epoch}.pth" for epoch in best_metric_epochs
    ]

    with TemporaryDirectory() as tmp_dir:
        checkpointer = CheckpointManager(
            logdir=tmp_dir,
            checkpoint_names=file_prefix,
            metric=metric_name,
            metric_minimization=minimize,
            save_n_best=n_best,
        )
        for epoch, metric in enumerate(metrics, start=1):
            checkpointer.process(
                metric_value=metric,
                epoch=epoch,
                checkpoint={"epoch": int(epoch), metric_name: float(metric)},
            )

        directory_files = os.listdir(tmp_dir)
        assert len(directory_files) == len(expected_files)
        for file in expected_files:
            assert file in directory_files

        for epoch, metric in zip(best_metric_epochs, best_metrics):
            content = torch.load(os.path.join(tmp_dir, f"experiment_{epoch}.pth"))
            assert all(k in content for k in ("epoch", metric_name))
            assert content["epoch"] == epoch
            assert content[metric_name] == metric


def test_checkpoint_manager_metric_minimization():
    n_best = np.random.randint(5, 20)
    minimize = True
    metric_name = "loss"
    file_prefix = "experiment"
    metrics = np.random.uniform(size=100)

    best_metric_epochs = np.argsort(metrics)
    if not minimize:
        best_metric_epochs = best_metric_epochs[::-1]
    best_metric_epochs = best_metric_epochs[:n_best]
    best_metrics = metrics[best_metric_epochs]
    best_metric_epochs += 1

    expected_files = ["metrics.json", "last.pth", "best.pth"] + [
        f"experiment_{epoch}.pth" for epoch in best_metric_epochs
    ]

    with TemporaryDirectory() as tmp_dir:
        checkpointer = CheckpointManager(
            logdir=tmp_dir,
            checkpoint_names=file_prefix,
            metric=metric_name,
            metric_minimization=minimize,
            save_n_best=n_best,
        )
        for epoch, metric in enumerate(metrics, start=1):
            checkpointer.process(
                metric_value=metric,
                epoch=epoch,
                checkpoint={"epoch": int(epoch), metric_name: float(metric)},
            )

        directory_files = os.listdir(tmp_dir)
        assert len(directory_files) == len(expected_files)
        for file in expected_files:
            assert file in directory_files

        for epoch, metric in zip(best_metric_epochs, best_metrics):
            content = torch.load(os.path.join(tmp_dir, f"experiment_{epoch}.pth"))
            assert all(k in content for k in ("epoch", metric_name))
            assert content["epoch"] == epoch
            assert content[metric_name] == metric

