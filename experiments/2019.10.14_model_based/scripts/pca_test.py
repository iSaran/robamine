from robamine.algo.util import EnvData, Dataset, Datapoint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

import pickle
import gym
import numpy as np

SEQUENCE_LENGTH = 1000

if __name__ == '__main__':
    data = EnvData.load('/home/iason/Dropbox/projects/phd/clutter/training/2019.10.14_model_based/transitions_pose/samples.env')

    dataset = Dataset()
    for i in range(len(data.info['extra_data'])):
        primitive = int(np.floor(data.transitions[i].action / 8))

        if primitive == 0:
            pos = data.info['extra_data'][i]['push_finger_vel']
            force = data.info['extra_data'][i]['push_finger_forces']
            poses = data.info['extra_data'][i]['target_object_displacement']

            # If the following is none it means we had a collision in Clutter
            # and no useful data from pushing was recorded
            if (pos is not None) and (force is not None):
                force = np.delete(force, 2, axis=1)
                # plt.plot(force)
                # plt.show()
                force_n = np.nan_to_num(force / np.linalg.norm(force, axis=1).reshape(-1, 1))
                # plt.plot(force)
                # plt.show()
                pos = np.delete(pos, 2, axis=1)
                vel = np.concatenate((np.zeros((1, 2)), np.diff(pos, axis=0)))
                vel_n = np.nan_to_num(vel / np.linalg.norm(vel, axis=1).reshape(-1, 1))

                inner = np.diag(np.matmul(-vel_n, np.transpose(force_n))).copy()

                # inputs = np.concatenate((pos, force), axis=1).ravel().copy()
                inputs = np.concatenate((force.ravel(), inner))
                # inputs = inner


                # inputs = np.delete(vel_force[1], 2, axis=1)
                # inputs = inputs[:-(inputs.shape[0] % SEQUENCE_LENGTH), :].ravel().copy()
                # outputs = poses[:-(poses.shape[0] % SEQUENCE_LENGTH), :][-1, :].ravel().copy()
                # if poses[-1, :][0] > 0 and poses[-1, :][1] > 0:
                #     label='++'
                # elif poses[-1, :][0] > 0 and poses[-1, :][1] < 0:
                #     label='+-'
                # elif poses[-1, :][0] < 0 and poses[-1, :][1] < 0:
                #     label='--'
                # elif poses[-1, :][0] < 0 and poses[-1, :][1] > 0:
                #     label='-+'
                if poses[-1, :][2] > 0:
                    label='+'
                else:
                # elif poses[-1, :][0] < 0:
                    label='-'
                dataset.append(Datapoint(x = inputs, y = label))

    x, y = dataset.to_array()
    x = StandardScaler().fit_transform(x)
    pca = PCA(.95)
    # pca = PCA(n_components=3)
    tsne = TSNE(n_components=3, verbose=1)
    pca_components = pca.fit_transform(x)
    # tsne_components = tsne.fit_transform(pca_components)

    print(pca.n_components_)
    print(pca_components.shape)
    print(pca.explained_variance_ratio_)
    components = pca_components

    fig = plt.figure()
    ax = Axes3D(fig)

    targets = ['+', '-']
    colors = ['r', 'g']

    # targets = ['++', '+-', '--', '-+']
    # colors = ['r', 'g', 'b', 'y']

    for target, color in zip(targets,colors):
        ax.scatter(components[np.where(y == target)][:, 0]
                   , components[np.where(y == target)][:, 1]
                   , components[np.where(y == target)][:, 2]
                   , c = color
                   , s = 2)
    plt.legend(targets)
    plt.grid()
    plt.show()
