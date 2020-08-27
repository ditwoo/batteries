
## Installing Pytorch XLA:

```bash
pip3 install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.6-cp36-cp36m-linux_x86_64.whl
```

## How to run:

```bash
PYTHONPATH=. python3 examples/device/experiment.py 2> /dev/null
```

## How to run tensorboard:

```bash
tensorboard --host 0.0.0.0 --logdir logs/ddp-experiment/tensorboard
```
