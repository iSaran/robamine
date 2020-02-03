import pickle
import os
import numpy as np
from robamine.algo.util import EnvData

def load_dataset(path):
    env_data = EnvData.load(os.path.join(path, 'samples.env'))
    n_samples = len(env_data.transitions)
    s = env_data.transitions[0].state.shape
    array = np.zeros((n_samples, s[0], s[1], s[2]))
    for i in range(n_samples):
        array[i, :, :, :] = env_data.transitions[i].state

if __name__ == '__main__':
    data = load_dataset('/home/iason/Dropbox/projects/phd/clutter/training/2020.02.03.autoencoder/robamine_logs_2020.02.03.17.25.16.362410')

    # train with data
    # ...
