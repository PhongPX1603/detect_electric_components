from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class Loss(Metric):
    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(Loss, self).__init__(output_transform)
        self._loss_fn = loss_fn

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        preds, targets = output
        # assert preds[0].shape[0] == targets[0].shape[0], 'the number of samples in predictions and targets must be equal.'
        losses = self._loss_fn(preds, targets)
        loss = losses[13] + losses[26] + losses[52]

        if len(loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        N = preds[0].shape[0]
        self._sum += loss.item() * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples
