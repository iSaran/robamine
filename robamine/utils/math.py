#!/usr/bin/env python3
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


def min_max_scale(x, range, target_range):
    assert range[1] > range[0]
    assert target_range[1] > target_range[0]

    if isinstance(x, np.ndarray):
        range_min = range[0] * np.ones(x.shape)
        range_max = range[1] * np.ones(x.shape)
        target_min = target_range[0] * np.ones(x.shape)
        target_max = target_range[1] * np.ones(x.shape)
    else:
        range_min = range[0]
        range_max = range[1]
        target_min = target_range[0]
        target_max = target_range[1]

    return target_min + ((x - range_min) * (target_max - target_min)) / (range_max - range_min)

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

class LineSegment2D:
    def __init__(self, p1, p2):
        self.p1 = p1.copy()
        self.p2 = p2.copy()

    def get_point(self, lambd):
        assert lambd >=0 and lambd <=1
        return (1 - lambd) * self.p1 + lambd * self.p2

    def get_lambda(self, p3):
        lambd = (p3[0] - self.p1[0]) / (self.p2[0] - self.p1[0])
        lambd_2 = (p3[1] - self.p1[1]) / (self.p2[1] - self.p1[1])
        if abs(lambd - lambd_2) > 1e-5:
            return None
        return lambd

    def get_intersection_point(self, line_segment, belong_self=True, belong_second=True):
        '''See wikipedia https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line'''
        x1 = self.p1[0]
        y1 = self.p1[1]
        x2 = self.p2[0]
        y2 = self.p2[1]

        x3 = line_segment.p1[0]
        y3 = line_segment.p1[1]
        x4 = line_segment.p2[0]
        y4 = line_segment.p2[1]

        if abs((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) < 1e-10:
            return None

        t =  ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / \
             ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        u = - ((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / \
              ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        p = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

        if belong_self and belong_second:
            if t >=0 and t <= 1 and u >= 0 and u <= 1:
                return p
            else:
                return None
        elif (belong_self and t >=0 and t <= 1) or (belong_second and u >= 0 and u <= 1):
            return p
        elif not belong_self and not belong_second:
            return p
        return None

    def norm(self):
        return np.linalg.norm(self.p1 - self.p2)

    def __str__(self):
        return self.p1.__str__() + self.p2.__str__()

    def array(self):
        result = np.zeros((2, 2))
        result[0, :] = self.p1
        result[1, :] = self.p2
        return result


    @staticmethod
    def plot_line_segments(line_segments, points=[]):
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(line_segments) + len(points))))
        lines = []
        i = 0
        for line_segment in line_segments:
            c = next(color)
            plt.plot(line_segment.p1[0], line_segment.p1[1], color=c, marker='o')
            plt.plot(line_segment.p2[0], line_segment.p2[1], color=c, marker='.')
            plt.plot([line_segment.p1[0], line_segment.p2[0]], [line_segment.p1[1], line_segment.p2[1]], color=c, linestyle='-')
            lines.append(Line2D([0], [0], label='LineSegment_' + str(i), color=c))
            i += 1

        i = 0
        for point in points:
            c = next(color)
            plt.plot(point[0], point[1], color=c, marker='o')
            lines.append(Line2D([0], [0], marker='o', label='Point_' + str(i), color=c))
            i += 1

        plt.legend(handles=lines)
        plt.show()


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

def triangle_area(t):
    """Calculates the area of a triangle defined given its 3 vertices. n_vertices x n_dims =  3 x 2"""
    return (1 / 2) * abs((t[0][0] - t[2][0]) * (t[1][1] - t[0][1]) - (t[0][0] - t[1][0]) * (t[2][1] - t[0][1]))
