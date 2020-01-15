from robamine.algo.util import NormalNoise
import matplotlib.pyplot as plt
import numpy as np

def run():
    noise = NormalNoise(mu=0.0, sigma=0.3)
    data = []
    for i in range(10000):
        data.append(noise())

    plt.hist(data, bins = 100)
    plt.show()

    noise = NormalNoise(mu=np.array([0.0, 0.0]), sigma=0.2)
    print(noise())


if __name__ == '__main__':
    run()
