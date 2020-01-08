from robamine.utils.memory import ReplayBuffer
from robamine.algo.util import Datapoint, Dataset
import numpy as np

import os
import pickle

DIR = '/home/iason/Dropbox/projects/phd/clutter/training/2019.10.14_model_based/yamls/robamine_logs_2019.10.14.14.58.03.472462'
n_substates = 8
n_primitives = 2

if __name__ == '__main__':
    #dataset = pickle.load(open(os.path.join(DIR, 'dataset.pkl'), 'rb'))
    #train, test = dataset['feature'].split(0.7)
    #print(type(train))
    #print(type(test))

    replay_buffer = ReplayBuffer.load(os.path.join(DIR, 'data.pkl'))
    pose = pickle.load(open(os.path.join(DIR, 'extra_data.pkl'), 'rb'))
    dataset = {'feature': [], 'pose': []}

    for i in range(n_primitives):
        dataset['feature'].append(Dataset())
        dataset['pose'].append(Dataset())

    # Create a data point for features:
    for i in range(replay_buffer.size()):
        state = np.split(replay_buffer(i).state, n_substates)
        next_state = np.split(replay_buffer(i).next_state, n_substates)
        primitive = int(np.floor(replay_buffer(i).action / n_substates))
        for j in range(n_substates):
            dataset['feature'][primitive].append(Datapoint(x = state[j], y = next_state[j]))
            dataset['pose'][primitive].append(Datapoint(x = state[j], y = pose[i]))

    pickle.dump(dataset, open(os.path.join(DIR, 'dataset.pkl'), 'wb'))
