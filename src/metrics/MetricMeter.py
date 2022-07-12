class MetricMeter:
    def __init__(self, meters, batch_size):
        self.meters = meters
        self.batch_size = batch_size

    def update(self, meter, value):
        self.meters[meter].update(value, self.batch_size)

    def get(self, meter):
        return self.meters[meter]

    def reset(self, meters):
        assert isinstance(meters, list), "meters must be a list"

        for meter in meters:
            self.meters[meter].reset()

    def add(self, meter):
        self.meters[meter] = AverageMeter()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

