import time
import json

import tensorflow as tf
import numpy as np


# https://stackoverflow.com/a/57303147
# https://github.com/tensorflow/tensorflow/blob/582c8d236cb079023657287c318ff26adb239002/tensorflow/python/keras/callbacks.py#L1858
class MetricBasedLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, verbose=0):
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        lr_new = self.schedule(epoch, lr, logs)

        if not tf.is_tensor(lr_new
                           ) and not isinstance(lr_new, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function ' 'should be float.')

        if tf.is_tensor(lr_new) and not lr_new.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')

        self.model.optimizer.lr.assign(lr_new)

        if tf.is_tensor(lr):
            lr = lr.numpy()

        logs = logs or {}
        logs['lr'] = lr

        if self.verbose > 0:
            print(
                '\nEpoch %05d: MetricBasedLearningRateScheduler set learning '
                'rate as %s.' % (epoch + 1, lr)
            )


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', mode='auto'):
        self.monitor = monitor

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs):
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs[self.monitor]
        if self.monitor_op(current_value, self.best):
            self.best = current_value
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs):
        self.model.set_weights(self.best_weights)


class TimeTracker(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.perf_counter()
        epoch_time = epoch_end_time - self.time
        self.time = epoch_end_time

        logs = logs or {}
        logs['time'] = epoch_time


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(
        self,
        monitor='val_loss',
        min_delta=0,
        patience=0,
        monitor_condition=lambda epoch, logs: True,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False
    ):
        super(CustomEarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.monitor_condition = monitor_condition
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            print('EarlyStopping mode %s is unknown, ' 'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            if self.monitor_condition(epoch, logs):
                self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        if self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s', self.monitor,
                ','.join(list(logs.keys()))
            )
        return monitor_value


class ModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, filepath, verbose=0):
        self.filepath = filepath
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        # filepath ending in '.h5' or '.keras' will default to HDF5 if save_format is None
        # otherwise save_format None defaults to 'tf'
        self.model.save_weights(filepath=self.filepath, overwrite=True, save_format=None)
        progress = {'epoch': epoch}
        try:
            with open(self.filepath + '.json', 'r+', encoding='utf-8') as f:
                prev_progress = json.load(f)
                progress['epoch'] = progress['epoch'] + int(prev_progress['epoch']) + 1
                json.dump(progress, f, ensure_ascii=False, indent=2)
        except IOError:
            pass
        else:
            with open(self.filepath + '.json', 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)

        if self.verbose > 0:
            print(
                '\nEpoch %05d: Saved model checkpoint to %s.' %
                (epoch + 1, self.filepath)
            )
