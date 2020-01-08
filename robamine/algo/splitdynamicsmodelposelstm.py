"""
Spit Dynamics Model Pose LSTM
=============================
"""

import torch
from robamine.algo.dynamicsmodel import LSTMSplitDynamicsModel
from robamine.algo.util import Datapoint, Dataset
from robamine.utils.math import rescale_array, filter_signal, Signal
import math
import numpy as np
from numpy.linalg import norm

import logging
logger = logging.getLogger('robamine.algo.splitdynamicsmodelposelstm')

SEQUENCE_LENGTH = 50

class SplitDynamicsModelPoseLSTM(LSTMSplitDynamicsModel):
    def __init__(self, params, inputs=4, outputs=3, name='SplitDynamicsModelPose'):
        super().__init__(params=params, inputs=inputs, outputs=outputs, name=name)

    def _preprocess_input(self, inputs, sequence_length):
        '''
        Preprocesses the inputs.

        inputs: a list with two ndarrays: pushfingervel and pushfingerforces
        returns: a list of preprocessed segments of inputs with length of sequence_length
        '''
        result = np.concatenate((np.delete(inputs[0], 2, axis=1), np.delete(inputs[1], 2, axis=1)), axis=1)
        result = result[:-(result.shape[0] % sequence_length), :]
        result = Signal(result).average_filter(SEQUENCE_LENGTH).array()
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

                outputs = poses[:-(poses.shape[0] % SEQUENCE_LENGTH), :]
                # Signal(outputs).plot()
                outputs = Signal(outputs).segment_last_element(SEQUENCE_LENGTH).array()

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

    def filter_datapoint(self, datapoint, epsilon=1e-8, filter=0.9, outliers_cutoff=3.8, plot=False):
        '''Assume datapoint is output of _clutter2array'''

        import matplotlib.pyplot as plt
        result = np.zeros(datapoint.shape)

        # Calculate force direction
        # -------------------------
        f = datapoint[:, 2:4].copy()
        f = np.nan_to_num(f / norm(f, axis=1).reshape(-1, 1))
        f_norm = norm(f, axis=1)

        # Find start and end of the contacts
        first = f_norm[0]
        for i in range(f_norm.shape[0]):
            if abs(f_norm[i] - first) > epsilon:
                break
        start_contact = i

        first = f_norm[-1]
        for i in reversed(range(f_norm.shape[0])):
            if abs(f_norm[i] - first) > epsilon:
                break;
        end_contact = i

        # No contact with the target detected
        if start_contact > end_contact:
            return result

        f = f[start_contact:end_contact, :]

        if plot:
            fig, axs = plt.subplots(2,2)
            axs[0][0].plot(f)
            plt.title('Force')
            axs[0][1].plot(norm(f, axis=1))
            plt.title('norm')

        f[:,0] = filter_signal(signal=f[:,0], filter=filter, outliers_cutoff=outliers_cutoff)
        f[:,1] = filter_signal(signal=f[:,1], filter=filter, outliers_cutoff=outliers_cutoff)
        f = np.nan_to_num(f / norm(f, axis=1).reshape(-1, 1))

        if plot:
            axs[1][0].plot(f)
            plt.title('Filtered force')
            axs[1][1].plot(norm(f, axis=1))
            plt.title('norm')
            plt.show()

        # Velocity direction
        p = datapoint[start_contact:end_contact, 0:2].copy()
        p_dot = np.concatenate((np.zeros((1, 2)), np.diff(p, axis=0)))
        p_dot_norm = norm(p_dot, axis=1).reshape(-1, 1)
        p_dot_normalized = np.nan_to_num(p_dot / p_dot_norm)

        if plot:
            fig, axs = plt.subplots(2)
            axs[0].plot(p_dot_normalized)
            axs[0].set_title('p_dot normalized')
            axs[1].plot(p_dot)
            axs[1].set_title('p_dot')
            plt.legend(['x', 'y'])
            plt.show()

        perpedicular_to_p_dot_normalized = np.zeros(p_dot_normalized.shape)
        for i in range(p_dot_normalized.shape[0]):
            perpedicular_to_p_dot_normalized[i, :] = np.cross(np.append(p_dot_normalized[i, :], 0), np.array([0, 0, 1]))[:2]

        inner = np.diag(np.matmul(-p_dot_normalized, np.transpose(f))).copy()
        inner_perpedicular = np.diag(np.matmul(perpedicular_to_p_dot_normalized, np.transpose(f))).copy()
        if plot:
            plt.plot(inner)
            plt.title('inner product')
            plt.show()

        result[start_contact:end_contact, 0:2] = p_dot_normalized
        result[start_contact:end_contact, 2:4] = f

        if plot:
            plt.plot(result)
            plt.show()

        return result
