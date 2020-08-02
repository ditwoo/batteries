import os
import json
from pathlib import Path
from queue import PriorityQueue
from collections import namedtuple
from typing import Union, Mapping, Any

# installed
import torch
import torch.nn as nn
import torch.optim as optim

# local files
from .utils import save_checkpoint


class CheckpointManager:
    def __init__(
        self,
        logdir: Union[str, Path],
        checkpoint_names: str = "exp",
        metric: str = "loss",
        metric_minimization: bool = True,
        save_n_best: int = 1,
    ):
        """
        Args:
            logdir (Union[str, Path]): directory where should be stored checkpoints
            checkpoint_names (str, optional): checkpoint file name.
                Default checkpoint name is "exp".
            metric (str, optional): metric name. Defaults to "loss".
            metric_minimization (bool, optional): indicator to minimize metric,
                if `True` then expected that target metric should decrease.
                Defaults to True.
            save_n_best (int, optional): number of best checkpoints to keep.
                Default is 1.
        """
        self.logdir = logdir
        self.checkpoint_filename = checkpoint_names
        self.metric_name = metric
        self.metric_minimization = metric_minimization
        self.save_n_best = save_n_best
        self.metrics = []
        self.best_metrics = []

    def _save_metrics(self) -> None:
        with open(os.path.join(self.logdir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

    def checkpoint_name(self, epoch: int) -> str:
        return f"{self.checkpoint_filename}_{epoch}.pth"

    def process(
        self, metric_value: float, epoch: int, checkpoint: Mapping[str, Any]
    ) -> None:
        """Generate checkpoint file and store only required checkpoints.

        Args:
            metric_value (float): value of a target metric
            epoch (int): epoch index
            checkpoint (Mapping[str, Any]): data to store in a checkpoint file
        """
        # determine arguments for save method
        if len(self.metrics):
            last_best_score = sorted(
                self.metrics,
                key=lambda record: record[self.metric_name],
                reverse=not self.metric_minimization,
            )[0][self.metric_name]
            if self.metric_minimization:
                is_best = metric_value <= last_best_score
            else:
                is_best = metric_value >= last_best_score
        else:
            is_best = True
        # store checkpoint
        checkpoint_name = self.checkpoint_name(epoch)
        save_checkpoint(
            checkpoint=checkpoint,
            logdir=self.logdir,
            name=checkpoint_name,
            is_best=is_best,
            is_last=True,
        )
        # update metrics
        metric_record = {"epoch": epoch, self.metric_name: metric_value}
        self.metrics.append(metric_record)
        self.best_metrics.append(metric_record)
        # remove old not required checkpoint
        if len(self.best_metrics) > self.save_n_best:
            self.best_metrics = sorted(
                self.best_metrics,
                key=lambda record: record[self.metric_name],
                reverse=not self.metric_minimization,
            )
            to_remove = os.path.join(
                self.logdir, self.checkpoint_name(self.best_metrics.pop(-1)["epoch"])
            )
            os.remove(to_remove)
        # overwrite existing metrics
        self._save_metrics()
