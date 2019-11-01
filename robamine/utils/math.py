#!/usr/bin/env python3
from math import exp
import numpy as np

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
