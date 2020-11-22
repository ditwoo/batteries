from pathlib import Path
from typing import Mapping, Union

from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    """Tensorboard wrapper.

    Args:
        logdir (str or pathlib.Path): directory where should be stored logs.
    """

    def __init__(self, logdir):  # noqa: D107
        self.writer = SummaryWriter(log_dir=logdir, max_queue=1)

    def __enter__(self):
        return self

    def metric(self, name, value, iteration):
        """Write metric using tensorboard writer.

        Args:
            name (str): name to use for metric values
            value (float or Dict[str, float]): metic to store,
                if passed dict with metics then they will be
                stored with name prefix.
            iteration (int): iteration number (batch/epoch index or similar)
        """
        if isinstance(value, (int, float)):
            self.writer.add_scalar(name, value, iteration)
        elif isinstance(value, dict):
            self.writer.add_scalars(name, value, iteration)

    def images(self, name, value, iteration):
        """Write images using tensorboard writer.

        Args:
            name (str): name to use for images
            value (np.ndarray or torch.tensor): tensor with images,
                should have shapes - (B)x(C)x(H)x(W)
            iteration (int): iteration number (batch/epoch index or similar)
        """
        self.writer.add_images(name, value, iteration)

    def __exit__(self, ex_type, ex_value, ex_traceback):
        self.writer.close()


if __name__ == "__main__":
    import shutil

    import numpy as np

    logdir = "./tb_check"
    shutil.rmtree(logdir, ignore_errors=True)

    with TensorboardLogger(logdir) as logger:
        x = np.arange(0, 10, step=0.001)
        for idx, _x in enumerate(x, start=1):
            logger.metric("y=x**2", {"x": _x, "y": _x ** 2}, idx)

        for idx, _x in enumerate(x[::10], start=1):
            logger.metric("y=x**3", {"x": _x, "y": _x ** 3}, idx)
