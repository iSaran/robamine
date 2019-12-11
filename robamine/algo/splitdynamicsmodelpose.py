"""
Spit Dynamics Model Pose
========================
"""

import torch
from robamine.algo.dynamicsmodel import FCSplitDynamicsModel
from robamine.algo.util import Datapoint, Dataset
from robamine.utils.math import rescale_array, Signal
import math
import numpy as np

import logging
logger = logging.getLogger('robamine.algo.splitdynamicsmodelpose')

SEQUENCE_LENGTH = 50

class SplitDynamicsModelPose(FCSplitDynamicsModel):
    def __init__(self, params, inputs=4, outputs=3, name='SplitDynamicsModelPose'):
        super().__init__(params=params, inputs=SEQUENCE_LENGTH*4, outputs=outputs, name=name)

    def _preprocess_input(self, inputs, sequence_length):
        '''
        Preprocesses the inputs.

        inputs: a list with two ndarrays: pushfingervel and pushfingerforces
        returns: a list of preprocessed segments of inputs with length of sequence_length
        '''
        result = np.concatenate((np.delete(inputs[0], 2, axis=1), np.delete(inputs[1], 2, axis=1)), axis=1)
        result = result[:-(result.shape[0] % sequence_length), :]
        result = Signal(result).average_filter(SEQUENCE_LENGTH).array().ravel().copy()
        return result

    def load_dataset(self, env_data):
        """ Transforms the EnvData x to a Dataset """

        # Create dataset
        # --------------

        dataset = []
        for i in range(self.nr_primitives):
            dataset.append(Dataset())

        for i in range(len(env_data.info['extra_data'])):
            primitive = int(np.floor(env_data.transitions[i].action / self.nr_substates))

            vel = env_data.info['extra_data'][i]['push_finger_vel']
            force = env_data.info['extra_data'][i]['push_finger_forces']
            poses = env_data.info['extra_data'][i]['target_object_displacement']

            # If the following is none it means we had a collision in Clutter
            # and no useful data from pushing was recorded
            if (vel is not None) and (force is not None):
                inputs = self._preprocess_input([vel, force], SEQUENCE_LENGTH)
                # Signal(inputs).plot()

                outputs = poses[:-(poses.shape[0] % SEQUENCE_LENGTH), :][-1, :].ravel().copy()

                dataset[primitive].append(Datapoint(x = inputs, y = outputs))

        for i in range(self.nr_primitives):
            if len(dataset[i]) > 0:
                self.dynamics_models[i].load_dataset(dataset[i])

    def predict(self, state, action):
        # Assuming that state is the pushing distance and action the index of
        # the action

        inputs = self._preprocess_input(state, SEQUENCE_LENGTH)
        prediction = np.array([0, 0, 0])
        primitive = int(np.floor(action / self.nr_substates))
        if primitive == 1:
            return prediction

        prediction = super().predict(inputs, action)
        return prediction

    def learn(self):
        self.iterations += 1

        for i in range(self.nr_primitives):
            if i!=1:
                self.dynamics_models[i].learn()
            self.info['train']['loss_' + str(i)] = \
                self.dynamics_models[i].info['train']['loss']
            self.info['test']['loss_' + str(i)] = \
                self.dynamics_models[i].info['test']['loss']
