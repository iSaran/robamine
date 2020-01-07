"""
Spit Dynamics Model
===================
"""

import torch
from robamine.algo.splitdynamicsmodel import SplitDynamicsModel
from robamine.algo.util import Datapoint, Dataset
from robamine.utils.math import rescale_array
import math
import numpy as np

import logging
logger = logging.getLogger('robamine.algo.splitdynamicsmodelpose')

class SplitDynamicsModelPose(SplitDynamicsModel):
    def __init__(self, params, inputs=2, outputs=3, name='SplitDynamicsModel'):
        super().__init__(params, inputs, outputs, name)

    def load_dataset(self, env_data):
        """ Transforms the EnvData x to a Dataset """

        # Create dataset
        # --------------

        dataset = []
        for i in range(self.nr_primitives):
            dataset.append(Dataset())

        for i in range(len(env_data.info['extra_data'])):
            # Extra data 0, 1 is the pushing distance and the angle of the action in rads
            pose_input = np.array([env_data.info['extra_data'][i][0], env_data.info['extra_data'][i][1]])
            primitive = int(np.floor(env_data.transitions[i].action / self.nr_substates))

            # Extra data in 2 has a np array with the displacement in (x, y, theta_around_z)
            dataset[primitive].append(Datapoint(x = pose_input, y = env_data.info['extra_data'][i][2]))

        for i in range(self.nr_primitives):
            self.dynamics_models[i].load_dataset(dataset[i])

    def predict(self, state, action):
        # Assuming that state is the pushing distance and action the index of
        # the action
        primitive = int(np.floor(action / self.nr_substates))
        substate_index = int(action - np.floor(action / self.nr_substates) * self.nr_substates)
        angle = substate_index * 2 * math.pi / (self.nr_primitives * self.nr_substates / self.nr_primitives)
        mystate = np.array([state, angle])
        prediction = super().predict(mystate, action)
        return prediction
