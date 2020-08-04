

## Kaggle competitions PyTorch training loop starter pack

## Generalized experiment flow

1. seed everything

2. create and initialize (probably load weights from checkpoint) model (sync batch norm and other), optimizer, scheduler

3. create datasets/dataloaders

4. train specified number of epochs

    4.1. update parameters on train set (do gradient accumulation & scheduler steps)

    4.2. compute metric on validation set (if need to compute metrics in epoch)

    4.3. log metrics with tensorboard or something similar

    4.4. generate checkpoint for model - store last state, best state (or few best states)

5. save the best score on validation set and compare it with score on leader board

### Examples

- [Training on specified device](examples/device)

- [Distributed training](examples/dp)

### Tests

```bash
make tests
```

### Removing unnecessary files

```bash
make clear
```