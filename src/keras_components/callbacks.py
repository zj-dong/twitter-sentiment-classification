import numpy as np

from keras import backend as K
from keras.callbacks import Callback


class CosineLRSchedule:
    """
    Cosine annealing with warm restarts, described in paper
    "SGDR: stochastic gradient descent with warm restarts"
    https://arxiv.org/abs/1608.03983

    Changes the learning rate, oscillating it between `lr_high` and `lr_low`.
    It takes `period` epochs for the learning rate to drop to its very minimum,
    after which it quickly returns back to `lr_high` (resets) and everything
    starts over again.

    With every reset:
        * the period grows, multiplied by factor `period_mult`
        * the maximum learning rate drops proportionally to `high_lr_mult`

    This class is supposed to be used with
    `keras.callbacks.LearningRateScheduler`.
    """
    def __init__(self, lr_high: float, lr_low: float, initial_period: int = 50,
                 period_mult: float = 2, high_lr_mult: float = 0.97):
        self._lr_high = lr_high
        self._lr_low = lr_low
        self._initial_period = initial_period
        self._period_mult = period_mult
        self._high_lr_mult = high_lr_mult

    def __call__(self, epoch, lr):
        return self.get_lr_for_epoch(epoch)

    def get_lr_for_epoch(self, epoch):
        assert epoch >= 0
        t_cur = 0
        lr_max = self._lr_high
        period = self._initial_period
        result = lr_max
        for i in range(epoch + 1):
            if i == epoch:  # last iteration
                result = (self._lr_low +
                          0.5 * (lr_max - self._lr_low) *
                          (1 + np.cos(np.pi * t_cur / period)))
            else:
                if t_cur == period:
                    period *= self._period_mult
                    lr_max *= self._high_lr_mult
                    t_cur = 0
                else:
                    t_cur += 1
        return result


class LinearLRScheduler(Callback):

    def __init__(self, lr_high, warmup_fraction, batch_size, total_epochs, data_size, print_every=None):
        super(LinearLRScheduler, self).__init__()
        self._lr_high = lr_high
        self._print_every = print_every

        self._total_steps = int((data_size / batch_size) * total_epochs)
        self._warmup_step_count = int(self._total_steps * warmup_fraction)
        self._current_step_count = 0
        self._warming_up = True

        self._lr_increase_per_step = self._lr_high / self._warmup_step_count
        self._lr_decrease_per_step = self._lr_high / (self._total_steps - self._warmup_step_count)
        self._lr_current = 0

    def on_batch_begin(self, batch, logs=None):
        assert batch >= 0
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        self._current_step_count += 1
        if self._warming_up:
            self._lr_current += self._lr_increase_per_step
            if self._current_step_count >= self._warmup_step_count:
                self._warming_up = False
        else:
            self._lr_current -= self._lr_decrease_per_step

        K.set_value(self.model.optimizer.lr, self._lr_current)

        if self._print_every and self._current_step_count % self._print_every == 0:
            print("Batch {:05d}: LinearLRScheduler setting learning rate to {}.".format(batch, self._lr_current))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class ProgressCallback(Callback):

    def __init__(self, logger):
        super(ProgressCallback, self).__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            logs = {}
        self.logger.info("Epoch {:04d}/{:04d} | acc: {:.3f}, val_acc: {:.3f}".format(
            epoch + 1, self.params["epochs"], logs.get("acc", np.nan), logs.get("val_acc", np.nan)))
