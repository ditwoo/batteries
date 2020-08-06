import os
import numpy as np
from tempfile import TemporaryDirectory

import torch
from batteries.checkpoint import (
    CheckpointManager,
    make_checkpoint,
    save_checkpoint,
    load_checkpoint,
)


def compare_state_dicts(a, b):
    for (ak, av), (bk, bv) in zip(a.items(), b.items()):
        if ak != bk:
            return False
        if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor):
            if not torch.equal(av, bv):
                return False
        else:
            if av != bv:
                return False
    return True


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


def test_make_checkpoint():
    stage = "some stage"
    epoch = 12345
    model = torch.nn.Linear(10, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 / 2)
    metrics = {"my metric": 0.123}

    checkpoint = make_checkpoint(stage, epoch, model, optimizer, metrics=metrics)

    print(model.state_dict())

    assert isinstance(checkpoint, dict)
    assert "stage" in checkpoint and checkpoint["stage"] == stage
    assert "epoch" in checkpoint and checkpoint["epoch"] == epoch
    assert "model_state_dict" in checkpoint and compare_state_dicts(
        checkpoint["model_state_dict"], model.state_dict()
    )
    assert "optimizer_state_dict" in checkpoint and compare_state_dicts(
        checkpoint["optimizer_state_dict"], optimizer.state_dict()
    )
    assert "scheduler_state_dict" not in checkpoint
    assert "metrics" in checkpoint and checkpoint["metrics"] == metrics


def test_save_checkpoint():
    checkpoint = {"some": "content"}
    with TemporaryDirectory() as tmp_dir:
        save_checkpoint(
            checkpoint, tmp_dir, "checkpoint", is_best=False, is_last=True,
        )
        assert os.path.isfile(os.path.join(tmp_dir, "checkpoint.pth"))
        assert not os.path.isfile(os.path.join(tmp_dir, "best.pth"))
        assert os.path.isfile(os.path.join(tmp_dir, "last.pth"))


def test_load_checkpoint():
    model = torch.nn.Linear(10, 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    checkpoint = make_checkpoint("stage", 1234, model, optimizer)

    model_from_checkpoint = torch.nn.Linear(10, 6)
    optimizer_from_checkpoint = torch.optim.Adam(model.parameters(), lr=1e-6)

    with TemporaryDirectory() as tmp_dir:
        save_checkpoint(
            checkpoint, tmp_dir, "checkpoint", is_best=False, is_last=True,
        )
        assert os.path.isfile(os.path.join(tmp_dir, "checkpoint.pth"))
        assert not os.path.isfile(os.path.join(tmp_dir, "best.pth"))
        assert os.path.isfile(os.path.join(tmp_dir, "last.pth"))

        load_checkpoint(
            os.path.join(tmp_dir, "checkpoint.pth"),
            model_from_checkpoint,
            optimizer_from_checkpoint,
        )

    assert compare_state_dicts(model.state_dict(), model_from_checkpoint.state_dict())
    assert compare_state_dicts(
        optimizer.state_dict(), optimizer_from_checkpoint.state_dict()
    )

