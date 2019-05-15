#!/usr/bin/env python3
from math import exp

def sigmoid(x, a=1, b=1, c=0, d=0):
    return a / (1 + exp(-b * x + c)) + d;
