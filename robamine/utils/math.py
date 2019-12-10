#!/usr/bin/env python3
from math import exp
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, a=1, b=1, c=0, d=0):
    return a / (1 + exp(-b * x + c)) + d;

def rescale_array(x, min=None, max=None, range=[0, 1], axis=None, reverse=False):
    assert range[1] > range[0]
    assert x.shape[0] > 1

    range_0 = range[0] * np.ones(x.shape)
    range_1 = range[1] * np.ones(x.shape)

    if min is None or max is None:
        if axis is None:
            _min = np.min(x) * np.ones(x.shape)
            _max = np.max(x) * np.ones(x.shape)
        else:
            _min = np.array([np.min(x, axis=axis),] * x.shape[0])
            _max = np.array([np.max(x, axis=axis),] * x.shape[0])
    else:
        _min = min * np.ones(x.shape)
        _max = max * np.ones(x.shape)

    if reverse:
        return _min + ((x - range_0) * (_max - _min)) / (range_1 - range_0)

    return range_0 + ((x - _min) * (range_1 - range_0)) / (_max - _min)


def rescale(x, min, max, range=[0, 1]):
    assert range[1] > range[0]
    return range[0] + ((x - min) * (range[1] - range[0])) / (max - min)

def filter_signal(signal, filter=0.9, outliers_cutoff=None):
    ''' Filters a signal

    Filters an 1-D signal using a first order filter and removes the outliers.

    filter: Btn 0 to 1. The higher the value the more the filtering.
    outliers_cutoff: How many times the std of the signal's diff away from the
    mean of the diff is a point considered outlier, typical value: 3.5. Set to
    None if you do not need this feature.
    '''
    signal_ = signal.copy()
    assert filter <= 1 and filter > 0

    if outliers_cutoff:
        mean_diff = np.mean(np.diff(signal_))
        std_diff = np.std(np.diff(signal_))
        lower_limit = mean_diff - std_diff * outliers_cutoff
        upper_limit = mean_diff + std_diff * outliers_cutoff

    for i in range(1, signal_.shape[0]):
        current_diff = signal_[i] - signal_[i-1]
        if outliers_cutoff and (current_diff > upper_limit or current_diff < lower_limit):
            filtering = 1
        else:
            filtering = filter

        signal_[i] = filtering * signal_[i - 1] + (1 - filtering) * signal_[i]

    return signal_

class Signal:
    def __init__(self, signal):
        self.signal = signal.copy()

    def average_filter(self, segments):
        assert self.signal.shape[0] % segments == 0
        splits = np.split(self.signal, segments, axis=0)
        result = np.concatenate([np.mean(segment, axis=0).reshape(1, -1) for segment in splits], axis=0)
        self.signal = result.copy()
        return self

    def segment_last_element(self, segments):
        assert self.signal.shape[0] % segments == 0
        splits = np.split(self.signal, segments, axis=0)
        result = np.concatenate([segment[-1, :].reshape(1, -1) for segment in splits], axis=0)
        self.signal = result.copy()
        return self

    def moving_average(self, n):
        ret = np.cumsum(self.signal, axis=0)
        ret[n:, :] = ret[n:, :] - ret[:-n, :]
        ret = ret[n - 1:] / n
        self.signal = ret.copy()
        return self

    def filter(self, a):
        '''
        a from 0 to 1, 1 means more filtering
        '''

        for i in range(1, self.signal.shape[0]):
            self.signal[i, :] = a * self.signal[i - 1, :] + (1 - a) * self.signal[i, :]

        return self

    def plot(self):
        plt.plot(self.signal)
        plt.show()

    def array(self):
        return self.signal
