import matplotlib.pyplot as plt
import numpy as np
import math
import yaml
from robamine.utils.math import rescale

def plot_epsilon():
    # Calculate epsilon for epsilon-greedy
    start = 0.9
    end = 0.05
    decay = 10000
    x = np.arange(0, decay * 5, 1)
    epsilon = []
    for i in x:
        epsilon.append(end + (start - end) * math.exp(-1 * i / decay))

    plt.plot(x, epsilon)
    plt.show()

def plot_exponential_reward():
    min_distance = -1
    max_distance = 1
    max_penalty = 10
    x = np.arange(min_distance, max_distance, 0.001)
    y = []
    for i in x:
        y.append(exp_reward(i, max_penalty, min_distance, max_distance))
    plt.plot(x,y)
    plt.show()

def exp_reward(x, max_penalty, min, max):
    a = 1
    b = -1.2
    c = -max_penalty
    min_exp = 0.0; max_exp = 5.0
    new_i = rescale(x, min, max, [min_exp, max_exp])
    return max_penalty * a * math.exp(b * new_i) + c


def plot_normal_noise():
    from robamine.algo.util import NormalNoise
    noise = NormalNoise(mu=0.0, sigma=0.2)
    data = []
    for i in range(10000):
        data.append(noise())

    plt.hist(data, bins = 100)
    plt.show()

    noise = NormalNoise(mu=np.array([0.0, 0.0]), sigma=0.2)
    print(noise())

if __name__ == '__main__':
    plot_exponential_reward()
