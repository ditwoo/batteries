from tensorboardX import SummaryWriter


class TensorboardLogger(SummaryWriter):  # noqa: D101
    def add_metric(self, tag, value, global_step=None, walltime=None):
        """Generalized method for `add_scalar` and `add_scalars`.
        If value is a dict with metrics (where key is metric value and value is a metric value)
        then will be called `add_scalars` othervise will be called `add_scalar`.

        Args:
            tag (str): name to use for metric values
            value (float or Dict[str, float/int]): metic to store,
                if passed dict with metics then they will be
                stored with name prefix.
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
                with seconds after epoch of event
        """
        if isinstance(value, dict):
            self.add_scalars(tag, value, global_step, walltime)
        else:
            self.add_scalar(tag, value, global_step, walltime)


if __name__ == "__main__":
    import shutil

    import numpy as np

    logdir = "./tb_check"
    shutil.rmtree(logdir, ignore_errors=True)

    with TensorboardLogger(logdir) as logger:
        x = np.arange(0, 10, step=0.001)
        for idx, _x in enumerate(x, start=1):
            logger.add_metric("y=x**2", {"x": _x, "y": _x**2}, idx)

        for idx, _x in enumerate(x[::10], start=1):
            logger.add_metric("y=x**3", {"x": _x, "y": _x**3}, idx)
