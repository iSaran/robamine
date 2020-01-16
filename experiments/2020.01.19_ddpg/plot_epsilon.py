import matplotlib.pyplot as plt
import numpy as np
import math
import yaml

def run():
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

if __name__ == '__main__':
    run()
