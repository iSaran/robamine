#!/usr/bin/env python3
from math import exp

def sigmoid(x, a=1, b=1, c=0, d=0):
    return a / (1 + exp(-b * x + c)) + d;

def rescale(x, min, max, range=[0, 1]):
    assert range[1] > range[0]
    return range[0] + ((x - min) * (range[1] - range[0])) / (max - min)
