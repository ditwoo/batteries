
## Batteries

<p align="center">
    <img src="https://image.flaticon.com/icons/svg/3103/3103476.svg" height="20%" width="20%">
</p>


## Generalized experiment flow

<ol>

<li> prepare environment, directory for logs, splits, datasets, augmentations (do a small research based on data).

```python
torch.autograd.set_detect_anomaly(False)  # small training speed improvement
torch.backends.cudnn.deterministic = True  # enabled reproducibility
torch.backends.cudnn.benchmark = False  # should be enabled for networks with fixed input & output sizes
# if you have issues with data loader workers
# (too much opened files)
torch.multiprocessing.set_sharing_strategy("file_system")
```

the most simple way to disable warnings:

```bash
PYTHONWARNINGS='ignore' python3 my_file.py
```

</li>

<li>
<details>
<summary>seed everything</summary> 

```python
import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

</details>
</li>

<li> create and initialize model, optimizer, scheduler

<ul>

<li>

<details>
<summary> load weights from checkpoint/pretrain or initialize with fixed seed </summary>
<p>

```python
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision.models import resnet18

from batteries import seed_all, load_checkpoint


# before model creation
seed_all(42)
model = resnet18()
# ...

# when load state from checkpoint
dist.barrier()
load_checkpoint("checkpoint.pth", model)
# ...

# fix seeds in workers
loader = DataLoader(
    dataset,
    # ...
    worker_init_fn=seed_all,
)
```

</p>
</details>
</li>

<li>

<details>
<summary> turn on sync batch norm for DDP setup </summary>
<p>

```python
import torch.nn as nn
from torchvision.models import resnet18

model = resnet18()
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
```

</p>
</details>

</li>

</ul>

<li> create datasets/dataloaders objects </li>

<li> train specified number of epochs

<ul>
<li>
<details>
<summary> update parameters on train set (do gradient accumulation & scheduler steps) </summary>
<p>

```python
for batch_index, (x, y) in enumerate(train_loader):
    x, y = x.cuda(), y.cuda()
    # set grads to zero
    optimizer.zero_grad(set_to_none=True)
    # retrieve outputs
    out = model(x)
    # compute loss on batch
    loss = loss_fn(out, y)
    # accumulate gradients
    loss.backward()
    # do weights update
    if (batch_index + 1) % accumulation_steps == 0:
        optimizer.step()
```

</p>
</details>

</li>

<li> compute metric on validation set </li>

<li> log metrics (use <b><a href="https://pytorch.org/docs/stable/tensorboard.html">tensorboard</a></b> / <b><a href="https://tensorboardx.readthedocs.io/en/latest/tensorboard.html">tensorboardX</a></b> / <b><a href="https://docs.wandb.ai/">wandb</a></b> / <b><a href="https://www.mlflow.org/docs/latest/index.html">mlflow</a></b>) </li>

<li>

<details>
<summary> generate checkpoints for model - store last state, best state (or few best states) </summary>
<p>


```python
from batteries import CheckpointManager

# ...
checkpointer = CheckpointManager(
    logdir=f"{logdir}/{stage}",
    metric=main_metric,
    metric_minimization=minimize_metric,
    save_n_best=5,
)

# ...
for epoch_index in range(1, n_epochs + 1):
    train_metrics = train_fn(...)
    valid_metrics = valid_fn(...)
    # main process will write weights to logdir
    if local_rank == 0:
        checkpointer.process(
            score=valid_metrics[main_metric],
            epoch=epoch_index,
            checkpoint=make_checkpoint(
                stage,
                epoch_index,
                model,
                optimizer,
                epoch_scheduler,
                metrics={"train": train_metrics, "valid": valid_metrics},
                experiment_args=args,
                model_args=model_args,
            ),
        )
```


</p>
</details>
</li>
</ul>

</li>

<li> save the best score on validation set and compare it with score on leader board </li>

<li> apply some postprocessing for submission - blend of scores (mean, power average, ranked average), SWA model checkpoints </li>

<li> check metric on test set and compare it with validation score </li>

<li> go to 1 and try to improve score on local validation and test set </li>

</ol>


## Experiment examples

<ol>

<li> <a href="examples/device"> Training on specified device </a> (multiple stages) </li>
<li> <a href="examples/dp"> Data Parallel training </a> (multiple stages) </li> 
<li>
<details>
<summary> Distributed Data Parallel minimal example </summary>

`experiment.py`:

```python
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from batteries import AverageMetter, CheckpointManager, get_logger, load_checkpoint, make_checkpoint, seed_all, t2d
from batteries.distributed import all_gather
from batteries.progress import tqdm

from datasets import MyDataset
from models import MyModel

logger = None


def setup(local_rank):
    """Initialize distributed experiment.

    Args:
        local_rank (int): process rank
    """
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", init_method="env://")

    world_size = torch.distributed.get_world_size()

    logger.info(
        "Training in distributed mode with multiple processes, 1 GPU per process. "
        f"Process {local_rank}, total {world_size}."
    )
    return device, world_size


def cleanup():
    """Close distributed experiment."""
    dist.destroy_process_group()


def get_loaders(batch_size, num_workers):
    """Build loaders for training.
    
    Args:
        batch_size (int): number of elements to use in train/valid data batches.
        num_workers (int): number of processes to use for generation batches.
    
    Returns:
        train and validation data loaders (torch.utils.data.DataLoader)
    """
    # TODO: finish train dataset
    train_dataset = ...
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_all,
        drop_last=True,
        sampler=train_sampler,
    )

    # TODO: finish validation dataset
    valid_dataset = ...
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        sampler=valid_loader,
    )

    return train_loader, valid_loader


def train_fn(model, loader, device, loss_fn, optimizer, scheduler=None, accumulation_steps=1, verbose=None):
    """One epoch training function.

    Args:
        model (torch.nn.Module): model to train.
        loader (torch.nn.utils.DataLoader): training data.
        device (int or str): rank of a device or device name to use for training.
        loss_fn (torch.nn.Module or function): function to compute a loss value.
        optimizer (torch.optim.Optimizer): model weights optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): batch learning rate scheduler.
            If `None` then will be ignored.
            Default is `None`.
        accumulation_steps (int): num steps to wait for performing backward pass.
            Default is `1`.
        verbose (float): log message with statistics after some % of iteration.

    Returns:
        dict with metrics collected during the training.
    """
    model.train()
    metrics = {"loss": AverageMetter()}
    num_batches = len(loader)

    for batch_index, batch in enumerate(loader):
        x, y = t2d(batch, device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)

        _loss = loss.item()

        loss.backward()

        if (batch_index + 1) % accumulation_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        metrics["loss"].update(_loss, x.size(0))

        if verbose and (batch_index + 1) % int(num_batches * verbose) == 0:
            logger.info("Train {} / {}: loss - {:.5f}".format(batch_index + 1, num_batches, metrics["loss"].average))

    return {"loss": metrics["loss"].average}


@torch.no_grad()
def valid_fn(model, loader, device, verbose=False):
    """Validate model on a given data.

    Args:
        model (torch.nn.Module): model to train.
        loader (torch.nn.utils.DataLoader): training data.
        device (int or str): rank of a device or device name to use for training.
        verbose (bool): option to show progress bar.
            Default is `False`.

    Returns:
        dict with metrics collected on a validation set.
    """
    model.eval()

    for batch_index, batch in enumerate(loader):
        x, y = t2d(batch, device)
        out = model(x)

    # compute scores & sync them with all_gather
    score = 0.0
    return {"metric": float(score)}


def log_metrics(stage, loader, epoch, metrics):
    """Write metrics using logger.

    Args:
        stage (str): stage name
        loader (str): loader name
        epoch (int): epoch number
        metrics (dict): metrics computed during training/validation steps
    """
    order = ("loss", "metric")
    metric_strs = []
    for metric_name in order:
        if metric_name in metrics:
            value = metrics[metric_name]
            metric_strs.append(f"{metric_name:>10} - {value:.4f}")
    logger.info(f"stage - {stage}, loader - {loader}, epoch - {epoch}: " + ",".join(metric_strs))


def experiment(local_rank, args=None):
    """Experiment entry point.

    Args:
        local_rank (int or str): device to use for training.
        world_size (ing): num devices in a world to use for training.
            If `None` then will be ignored.
            Default is `None`.
        args (Dict[str, Any]): experiment arguments.
            Default is `None`.
    """
    main_metric = "metric"
    minimize_metric = False

    args = {} if args is None else args
    logdir = args["logdir"]
    exp_name = args["exp_name"]

    # create logdir if not exists
    if not os.path.isdir(logdir) and local_rank == 0:
        os.makedirs(logdir)

    global logger
    logger = get_logger(exp_name, log_file=os.path.join(logdir, "train.log"), disable=(local_rank == 0))

    device, world_size = setup(local_rank)

    if local_rank == 0:
        wandb.init(project=exp_name)

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Experiment arguments: {args}")
    logger.info(f"Main metric - '{main_metric}'")
    logger.info(f"Minimize metric - '{minimize_metric}'")

    train_loader, valid_loader, tokenizer = get_loaders(args["bs"], args["workers"])

    seed_all(42)
    model_args = {}  # TODO: use your own args
    model = MyModel(**model_args)
    dist.barrier()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    optimizer_args = dict(lr=1e-3, weight_decay=1e-6)
    optimizer = optim.AdamW(model.module.parameters(), **optimizer_args)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    batch_scheduler = None
    epoch_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args["epochs"])

    if args["continue"]:
        dist.barrier()
        load_checkpoint(args["continue"], model)

    logger.info("Model: MyModel")
    logger.info(f"Model args: {model_args}")
    logger.info("Optimizer: AdamW")
    logger.info(f"Optimizer args: {optimizer_args}")

    stage = "stage_0"
    n_epochs = args["epochs"]

    checkpointer = CheckpointManager(
        logdir=os.path.join(logdir, stage),
        metric=main_metric,
        metric_minimization=minimize_metric,
        save_n_best=5,
    )

    for epoch_index in range(1, n_epochs + 1):
        logger.info(f"Epoch {epoch_index}/{n_epochs}")

        if train_loader.sampler and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch_index)

        train_metrics = train_fn(model, train_loader, device, criterion, optimizer, batch_scheduler, verbose=0.01)
        log_metrics(stage, "train", epoch_index, train_metrics)

        valid_metrics = valid_fn(model, valid_loader, device, tokenizer, verbose=args["progress"])
        log_metrics(stage, "valid", epoch_index, valid_metrics)

        if local_rank == 0:
            checkpointer.process(
                score=valid_metrics[main_metric],
                epoch=epoch_index,
                checkpoint=make_checkpoint(
                    stage,
                    epoch_index,
                    model,
                    optimizer,
                    epoch_scheduler,
                    metrics={"train": train_metrics, "valid": valid_metrics},
                    experiment_args=args,
                    model_args=model_args,
                ),
            )

        if epoch_scheduler is not None:
            epoch_scheduler.step()

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--exp-name", dest="exp_name", type=str, help="experiment name", default="experiment")
    parser.add_argument("--fold", dest="fold", type=int, help="fold index to use as validation", default=0)
    parser.add_argument("--batch-size", dest="bs", type=int, help="batch size", default=2)
    parser.add_argument("--num-workers", dest="workers", type=int, help="number of workers to use", default=1)
    parser.add_argument("--num-epochs", dest="epochs", type=int, help="number of epochs to train", default=1)
    parser.add_argument(
        "--logdir", dest="logdir", type=str, help="directory where should be stored logs", default="logs/test"
    )
    parser.add_argument(
        "--continue", dest="continue", type=str, help="checkpoint to use for pretrained model", default=None
    )
    # put here additional arguments
    # ...
    
    args = vars(parser.parse_args())
    local_rank = args["local_rank"]
    experiment(local_rank, args)


if __name__ == "__main__":
    main()

```

to run:

```bash

LOGDIR=./logs/my_experiment

if [[ -d ${LOGDIR} ]]
then
    rm -rf ${LOGDIR};
    echo "[!] Removed existing directory with logs ('${LOGDIR}')!";
    mkdir -p ${LOGDIR};
fi

PYTHONPATH=.:${PYTHONPATH} \
MASTER_PORT=12345 \
python3 -m torch.distributed.launch --nproc_per_node=2 \
    experiment.py \
    --exp-name='my experiment' \
    --fold=0 \
    --batch-size=128 \
    --num-workers=32 \
    --num-epochs=1234 \
    --progress \
    --logdir=${LOGDIR}
```

</details>
</li>

</ol>

Good example of training script - [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/master/train.py)


