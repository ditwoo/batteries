

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


1. seed everything

2. create and initialize model, optimizer, scheduler

    2.1. load weights from checkpoint/pretrain or initialize (not randomly with fixed seed!)

    2.2. if [`DistributedDataParallel`](https://pytorch.org/docs/stable/notes/ddp.html) - turn on sync batch norm

3. create datasets/dataloaders objects

4. train specified number of epochs

    4.1. update parameters on train set (do gradient accumulation & scheduler steps)

    4.2. compute metric on validation set (if need to compute metrics in epoch)

    4.3. log metrics (use tensorboard or something similar)

    4.4. generate checkpoints for model - store last state, best state (or few best states)

5. save the best score on validation set and compare it with score on leader board

6. apply some postprocessing for submission - blend of scores (mean, power average, ranked average), SWA model checkpoints

7. check metric on test set and compare it with validation score

8. go to 0 and try to improve score on local validation and test set


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
