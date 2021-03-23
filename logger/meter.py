class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.
        self._count = 0

    def update(self, val, n=1):
        self._sum += val * n
        self._count += n

    @property
    def avg(self):
        return self._sum / self._count
