"""
Spit Dynamics Model Pose
========================
"""

import torch
from robamine.algo.dynamicsmodel import FCSplitDynamicsModel
from robamine.algo.util import Datapoint, Dataset
from robamine.utils.math import rescale_array
import math
import numpy as np

import logging
logger = logging.getLogger('robamine.algo.splitdynamicsmodelpose')

class SplitDynamicsModelPose(FCSplitDynamicsModel):
    def __init__(self, params, inputs=4, outputs=3, name='SplitDynamicsModelPose'):
        super().__init__(params=params, inputs=inputs, outputs=outputs, name=name)

    def load_dataset(self, env_data):
        """ Transforms the EnvData x to a Dataset """

        # Create dataset
        # --------------

        dataset = []
        for i in range(self.nr_primitives):
            dataset.append(Dataset())

        for i in range(len(env_data.info['extra_data'])):
            # Extra data 0, 1 is the pushing distance and the angle of the action in rads

            j = int(env_data.transitions[i].action - np.floor(env_data.transitions[i].action / self.nr_substates) * self.nr_substates)
            state_split = np.split(env_data.transitions[i].state, self.nr_substates)
            push_distance = state_split[j][-2]
            bounding_box_1 = state_split[j][-7]
            bounding_box_2 = state_split[j][-8]
            angle = j * 2 * math.pi / (self.nr_primitives * self.nr_substates / self.nr_primitives)
            pose_input = np.array([push_distance, bounding_box_1, bounding_box_2, angle])
            primitive = int(np.floor(env_data.transitions[i].action / self.nr_substates))

            # Extra data in 2 has a np array with the displacement in (x, y, theta_around_z)
            dataset[primitive].append(Datapoint(x = pose_input, y = env_data.info['extra_data'][i]['displacement'][2]))

        for i in range(self.nr_primitives):
            self.dynamics_models[i].load_dataset(dataset[i])

    def predict(self, state, action):
        # Assuming that state is the pushing distance and action the index of
        # the action
        primitive = int(np.floor(action / self.nr_substates))
        if primitive == 1:
            return np.array([0, 0, 0])
        substate_index = int(action - np.floor(action / self.nr_substates) * self.nr_substates)
        angle = substate_index * 2 * math.pi / (self.nr_primitives * self.nr_substates / self.nr_primitives)
        mystate = np.array([state[0], state[1], state[2], angle])
        prediction = super().predict(mystate, action)
        return prediction
