

## Batteries

<p align="center">
    <img src="https://image.flaticon.com/icons/svg/3103/3103476.svg" height="20%" width="20%">
</p>

## Generalized experiment flow

0. Prepare environment, logdir, splits, datasets, augmentations (do a small research based on data).

1. seed everything

2. create and initialize (probably load weights from checkpoint) model (sync batch norm and other), optimizer, scheduler

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

- [Training on specified device](examples/device)

- [Data Parallel training](examples/dp)

- [Distributed Data Parallel training](examples/ddp)

### Tests

```bash
make tests
```

### Removing unnecessary files

```bash
make clear
```