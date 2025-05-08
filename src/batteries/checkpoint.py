import json
import os
import shutil
import warnings
from collections import OrderedDict

import torch


def make_checkpoint(stage, epoch, model, optimizer=None, scheduler=None, metrics=None, **kwargs) -> dict:
    """Generate checkpoint dict.

    Args:
        stage (str): stage name
        epoch (int): epoch index
        model (torch.nn.Module or torch.nn.DataParallel): model
        optimizer (torch.optim.Optimizer): optimizer.
            Default is ``None``.
        scheduler (torch.optim.lr_scheduler._LRScheduler): scheduler.
            Default is ``None``.
        metrics (dict, optional): metrics to store in checkpoint.
            Default is ``None``.
        **kwargs: other keys with values to store in checkpoint.

    Returns:
        dict: [description]
    """
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return make_checkpoint(stage, epoch, model.module, optimizer, scheduler, metrics)

    if not isinstance(model, torch.nn.Module):
        raise ValueError("Expected that model will be an instance of nn.Module but got {}!".format(type(model)))

    checkpoint = {"stage": stage, "epoch": epoch}
    if model is not None:
        checkpoint["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if metrics is not None:
        checkpoint["metrics"] = metrics

    for key, value in kwargs.items():
        if key in checkpoint:
            warnings.warn(f"Found duplicated keyword ('{key}'), it will be ignored!")
            continue
        checkpoint[key] = value

    return checkpoint


def save_checkpoint(
    checkpoint,
    logdir,
    name,
    is_best=False,
    is_last=False,
    verbose=False,
    save_fn=torch.save,
) -> None:
    """Save checkpoint to a file.

    Args:
        checkpoint (dict): data to store in checkpoint
        logdir (str or Path): directory where should be stored checkpoint
        name (str): file name to use for storing checkpoint
        is_best (bool, optional): indicator to save checkpoint as best checkpoint.
            Defaults to False.
        is_last (bool, optional): indicator to save checkpoint as last checkpoint.
            Defaults to False.
        verbose (bool, optional): default is `False`.
        save_fn (function (callable), optional): default is `torch.save`
    """
    os.makedirs(logdir, exist_ok=True)
    _name = name if name.endswith(".pth") else f"{name}.pth"
    filename = os.path.join(str(logdir), _name)
    save_fn(checkpoint, filename)
    if verbose:
        print(f"=> Saved checkpoint '{filename}'")
    if is_best:
        best_filename = os.path.join(str(logdir), "best.pth")
        shutil.copyfile(filename, best_filename)
    if is_last:
        last_filename = os.path.join(str(logdir), "last.pth")
        shutil.copyfile(filename, last_filename)


def load_checkpoint(
    checkpoint_file,
    model,
    optimizer=None,
    scheduler=None,
    map_location=None,
    verbose=True,
):
    """Shortcut for loading checkpoint state.

    Args:
        checkpoint_file (str or Path): path to checkpoint.
        model (torch.nn.Module): model to initialize with checkpoint weights
        optimizer (torch.optim.Optimizer): optimizer to initialize with checkpoint weights.
            If `None` then will be ignored.
            Default is `None`.
        scheduler (torch.optim.lr_scheduler._LRScheduler): scheduler to initialize with checkpoint weights.
            If `None` then will be ignored.
            Default is `None`.
        map_location (torch.device or str or dict[str, int]):
            location to use for loading checkpoint content.
            More about possible locations - `https://pytorch.org/docs/master/generated/torch.load.html`
            Default is `None`.
        verbose (bool): verbosity mode, if `True` then will print a loaded items.
            Default is `True`.
    """  # noqa: D417
    checkpoint = torch.load(str(checkpoint_file), map_location=map_location)
    loaded_items = []

    if "model_state_dict" in checkpoint and model is not None:
        state_dict = checkpoint["model_state_dict"]
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        loaded_items.append("model")

    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loaded_items.append("optimizer")

    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        loaded_items.append("scheduler")

    if loaded_items and verbose:
        print("<= Loaded {} from '{}'".format(", ".join(loaded_items), checkpoint_file))

        if "stage" in checkpoint:
            print("Stage: {}".format(checkpoint["stage"]))

        if "epoch" in checkpoint:
            print("Epoch: {}".format(checkpoint["epoch"]))

        if "metrics" in checkpoint:
            print("Metrics:")
            print(checkpoint["metrics"])


def average_model_state_dicts(*files) -> OrderedDict:
    """Compute average model state from files.

    Args:
        files: path to checkpoint files (should be present 'model_state_dict').

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
            map_location=(lambda s, _: torch.serialization.default_restore_location(s, "cpu")),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        if "model_state_dict" not in state:
            raise KeyError(f"Missing 'model_state_dict' in '{f}'!")

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
        if v.dtype in (torch.short, torch.int, torch.long):
            averaged_params[k] = v // num_models
        else:
            averaged_params[k] = v / num_models
    new_state["model_state_dict"] = averaged_params
    return new_state


class CheckpointManager:
    """Manage saving top N best checkpoints based on metric.

    Args:
        logdir (str or Path): directory where should be stored checkpoints
        checkpoint_names (str, optional): checkpoint file name.
            Default checkpoint name is "exp".
        metric (str, optional): metric name.
            Default is "loss".
        metric_minimization (bool, optional): indicator to minimize metric,
            if `True` then expected that target metric should decrease.
            Default is True.
        save_n_best (int, optional): number of best checkpoints to keep.
            Default is 1.
        save_fn (function (callable), optional): model save function.
            Default is `torch.save`.
        metrics_file (str): file to use for storing metrics.
    """

    def __init__(
        self,
        logdir,
        checkpoint_names="exp",
        metric="loss",
        metric_minimization=True,
        save_n_best=1,
        save_fn=torch.save,
        metrics_file="metrics.json",
    ):  # noqa: D107
        self.logdir = logdir
        self.checkpoint_filename = checkpoint_names
        self.metric_name = metric
        self.metric_minimization = metric_minimization
        self.save_n_best = save_n_best
        self.metrics = []  # list of dicts where 2 keys required - metric_name & 'epoch'
        self.best_metrics = []
        self.save_fn = save_fn
        self.metrics_file = metrics_file if metrics_file.endswith(".json") else f"{metrics_file}.json"

    def __repr__(self):  # noqa: D105
        return (
            "CheckpointManager("
            f"logdir={self.logdir},"
            f"checkpoint_names={self.checkpoint_filename},"
            f"metric={self.metric_name},"
            f"metric_minimization={self.metric_minimization},"
            f"save_n_best={self.save_n_best},"
            f"save_fn={self.save_fn},"
            f"metrics_file={self.metrics_file}"
            ")"
        )

    def _save_metrics(self) -> None:
        """Store checkpoint information to a file."""
        to_save = {
            "metric_name": self.metric_name,
            "metric_minimization": self.metric_minimization,
            "values": self.metrics,
        }
        file_path = os.path.join(self.logdir, self.metrics_file)
        with open(file_path, "w") as f:
            json.dump(to_save, f, indent=4)

    def _checkpoint_name(self, epoch) -> str:
        """Get checkpoint file name.

        Args:
            epoch (int): epoch number

        Returns:
            string with checkpoint name
        """
        return f"{self.checkpoint_filename}_{epoch}.pth"

    def process(self, score, epoch, checkpoint) -> None:
        """Generate checkpoint file and store only required checkpoints.

        Args:
            score (float or Dict[str, float]): target metric value
                or dict with metric_name key.
            epoch (int): epoch index
            checkpoint (Dict[str, Any]): data to store in a checkpoint file
        """
        # unpack metric value
        if isinstance(score, dict):
            if self.metric_name not in score:
                raise KeyError(f"There is no '{self.metric_name}' in {score}!")
            _metric = score[self.metric_name]
        else:
            _metric = score

        # collect arguments for save method
        if len(self.metrics):
            last_best_score = sorted(
                self.metrics,
                key=lambda record: record[self.metric_name],
                reverse=not self.metric_minimization,
            )[0][self.metric_name]
            if self.metric_minimization:
                is_best = _metric <= last_best_score
            else:
                is_best = _metric >= last_best_score
        else:
            is_best = True

        # store checkpoint
        checkpoint_name = self._checkpoint_name(epoch)
        save_checkpoint(
            checkpoint=checkpoint,
            logdir=self.logdir,
            name=checkpoint_name,
            is_best=is_best,
            is_last=True,
            save_fn=self.save_fn,
        )

        # update metrics
        metric_record = dict(score) if isinstance(score, dict) else {}
        metric_record["epoch"] = epoch
        metric_record[self.metric_name] = _metric

        self.metrics.append(metric_record)
        self.best_metrics.append(metric_record)
        # remove old not required checkpoint
        if len(self.best_metrics) > self.save_n_best:
            self.best_metrics = sorted(
                self.best_metrics,
                key=lambda record: record[self.metric_name],
                reverse=not self.metric_minimization,
            )
            to_remove = os.path.join(self.logdir, self._checkpoint_name(self.best_metrics.pop(-1)["epoch"]))
            try:
                os.remove(to_remove)
            except FileNotFoundError:
                pass

        # overwrite existing metrics
        self._save_metrics()
