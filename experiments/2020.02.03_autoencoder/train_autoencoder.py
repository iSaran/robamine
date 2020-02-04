import pickle
import os
import numpy as np
from robamine.algo.util import EnvData

from robamine.algo.util import AutoEncoder

def load_dataset(path):
    env_data = EnvData.load(os.path.join(path, 'samples.env'))
    n_samples = len(env_data.transitions)
    s = env_data.transitions[0].state.shape
    array = np.zeros((n_samples, s[0], s[1], s[2]))
    for i in range(n_samples):
        array[i, :, :, :] = env_data.transitions[i].state
    return array

if __name__ == '__main__':
    data = load_dataset('/home/mkiatos/robamine/logs/robamine_logs_2020.02.03.18.36.16.854111/')
    print(data.shape)

    ae = AutoEncoder()

    # train with data
    # ...
