from pathlib import Path
from typing import Union, Mapping
from tensorboardX import SummaryWriter


class TensorboardLogger:
    def __init__(self, logdir: Union[str, Path]):
        self.writer = SummaryWriter(log_dir=logdir)

    def __enter__(self):
        return self

    def metric(
        self, plot_name: str, value: Union[float, Mapping[str, float]], iteration: int
    ) -> None:
        self.writer.add_scalars(plot_name, value, iteration)

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
