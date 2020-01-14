from robamine.algo.util import EnvData, Dataset, Datapoint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pickle
import gym
import numpy as np

SEQUENCE_LENGTH = 1000

if __name__ == '__main__':
    dataset = Dataset()
    for i in range(10000):
        f = np.zeros(1000)
        sin_or_cos = np.random.randint(2)
        if sin_or_cos == 0:
            function = np.sin
        else:
            function = np.cos
        inputs = function(2 * np.pi * (1/200) * np.arange(200)) + np.random.normal(loc=0.0, scale=0.1, size=(200,))
        start_index = np.random.randint(1000)
        end_index = start_index + 200

        missing = end_index - 1000
        if missing > 0:
            end_index = 1000

        f[start_index:end_index] = inputs[0:200-missing]

        # plt.plot(f)
        # plt.show()

        if sin_or_cos == 0:
            label = 'sin'
        else:
            label = 'cos'
        dataset.append(Datapoint(x = f, y = label))

    x, y = dataset.to_array()
    x = StandardScaler().fit_transform(x)
    # pca = PCA(.95)
    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    print(components)
    print(pca.n_components_)
    print(type(components))
    print(components.shape)
    print(pca.explained_variance_ratio_)

    fig = plt.figure()
    # ax = Axes3D(fig)

    targets = ['sin', 'cos']
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        plt.scatter(components[np.where(y == target)][:, 0]
                   , components[np.where(y == target)][:, 1]
                   # , components[np.where(y == target)][:, 2]
                   , c = color
                   , s = 50)
    plt.legend(targets)
    plt.grid()
    plt.show()


# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2'])
