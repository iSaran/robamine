from robamine.utils.math import sigmoid
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    lin = [-1, 1]
    x = np.linspace(lin[0], lin[1], 1000)
    y = []
    for xx in x:
        y.append(sigmoid(xx, a=-5, b=-15/0.5, c=-4))
    plt.plot(x, y)
    print(sigmoid(0.1, a=-5, b=-20/0.25, c=-4))
    plt.axis([0, 0.5, -6, 1])
    plt.show()
