
"""
Spit Dynamics Model
===================
"""

import torch
from robamine.algo.dynamicsmodel import FCSplitDynamicsModel
from robamine.algo.util import Datapoint, Dataset
from robamine.utils.math import rescale_array
import math
import numpy as np

import logging
logger = logging.getLogger('robamine.algo.splitdynamicsmodelfeature')

class SplitDynamicsModelFeature(FCSplitDynamicsModel):
    def __init__(self,  params, inputs=None, outputs=None, name='SplitDynamicsModelFeature'):
        nr_primitives = int(len(params['hidden_units']))
        nr_substates = int(params['action_dim'] / nr_primitives)
        dim = int(params['state_dim'] / nr_substates)
        super().__init__(params=params, inputs=dim, outputs=dim, name=name)

    def load_dataset(self, env_data):
        """ Transforms the EnvData x to a Dataset """

        # Create dataset
        # --------------

        dataset = []
        for i in range(self.nr_primitives):
            dataset.append(Dataset())

        for i in range(len(env_data.transitions)):
            state = np.split(env_data.transitions[i].state, self.nr_substates)
            next_state = np.split(env_data.transitions[i].next_state, self.nr_substates)
            primitive = int(np.floor(env_data.transitions[i].action / self.nr_substates))

            for j in range(self.nr_substates):
                dataset[primitive].append(Datapoint(x = state[j], y = next_state[j]))

        for i in range(self.nr_primitives):
            self.dynamics_models[i].load_dataset(dataset[i])

    def predict(self, state, action):
        # Assuming that state is the pushing distance and action the index of
        # the action
        primitive = int(np.floor(action / self.nr_substates))
        substate_index = int(action - np.floor(action / self.nr_substates) * self.nr_substates)
        features = np.split(state, self.nr_substates)

        for j in range(0, self.nr_substates):
            next_feature = self.dynamics_models[primitive].predict(features[j])

            if j == 0:
                prediction = next_feature
            else:
                prediction = np.concatenate((prediction, next_feature))

        return prediction
