# noqa: D401


class AverageMetter:
    """Compute and store average value of metric."""

    def __init__(self):  # noqa: D107
        self.reset()

    def reset(self):
        """Resets internal values to a default state."""
        self.value = 0.0
        self.average = 0.0
        self.sum = 0.0
        self.counter = 0

    def update(self, value, times=1):
        """Add value to metter.

        Args:
            value (int or float): value to store.
            times (int): number of times to add value to the store.
                Default is `1`.
        """
        self.value = value
        self.sum += value * times
        self.counter += times
        self.average = self.sum / self.counter

    def __repr__(self):  # noqa: D105
        return f"AverageMetter(value={self.value}," f"average={self.average},sum={self.sum}," f"counter={self.counter})"
