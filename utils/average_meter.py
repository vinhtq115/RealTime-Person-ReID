class AverageMeter(object):
    """Used to keep track of the average of a set of values.
    Modified from PyTorch's implementation."""

    def __init__(self):
        """Initialize the AverageMeter class."""
        self.reset()

    def reset(self):
        """Reset all values to 0."""
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0
