

## Batteries

<p align="center">
    <img src="https://image.flaticon.com/icons/svg/3103/3103476.svg" height="20%" width="20%">
</p>

## Generalized experiment flow

0. prepare environment, directory for logs, splits, datasets, augmentations (do a small research based on data).

    ```python
    # pytorch speed hacks
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if you have issues with data loader workers
    # (too much opened files)
    torch.multiprocessing.set_sharing_strategy("file_system")
    ```

    the most simple way to disable warnings:

    ```bash
    PYTHONWARNINGS='ignore' python3 my_file.py
    ```

<ol>

<li> seed everything </li>

<li> create and initialize model, optimizer, scheduler

<ul>

<li>

<details>
<summary> load weights from checkpoint/pretrain or initialize (not randomly with fixed seed!) </summary>
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
<summary> turn on sync batch norm for DDP setup: </summary>
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

<li> generate checkpoints for model - store last state, best state (or few best states) </li>
</ul>

</li>

<li> save the best score on validation set and compare it with score on leader board </li>

<li> apply some postprocessing for submission - blend of scores (mean, power average, ranked average), SWA model checkpoints </li>

<li> check metric on test set and compare it with validation score </li>

<li> go to 1 and try to improve score on local validation and test set </li>

</ol>

### Experiment examples

- [Training on specified device](examples/device) (multiple stages)
- [Data Parallel training](examples/dp) (multiple stages)
- [Distributed Data Parallel training](examples/ddp)

Good example of training script - [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/master/train.py)

### Tests

```bash
make tests
```

### Removing unnecessary files

```bash
make clear
```
