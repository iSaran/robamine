"""
Spit Dynamics Model Pose LSTM
=============================
"""

import torch
from robamine.algo.dynamicsmodel import LSTMSplitDynamicsModel
from robamine.algo.util import Datapoint, Dataset
from robamine.utils.math import rescale_array
import math
import numpy as np

import logging
logger = logging.getLogger('robamine.algo.splitdynamicsmodelposelstm')

class SplitDynamicsModelPoseLSTM(LSTMSplitDynamicsModel):
    def __init__(self, params, inputs=4, outputs=3, name='SplitDynamicsModelPose'):
        super().__init__(params=params, inputs=inputs, outputs=outputs, name=name)

    def _clutter2array(self, extra_data):
        inputs = np.concatenate((extra_data[0], extra_data[1]), axis=1)
        inputs = np.delete(inputs, 2, axis=1)
        inputs = np.delete(inputs, -1, axis=1)
        return inputs

    def load_dataset(self, env_data):
        """ Transforms the EnvData x to a Dataset """

        # Create dataset
        # --------------

        dataset = []
        for i in range(self.nr_primitives):
            dataset.append(Dataset())

        for i in range(len(env_data.info['extra_data'])):
            # Extra data 0, 1 is the pushing distance and the angle of the action in rads

            primitive = int(np.floor(env_data.transitions[i].action / self.nr_substates))

            # print(env_data.info['extra_data'][i]['push_forces_vel'])
            inputs = self._clutter2array(env_data.info['extra_data'][i]['push_forces_vel'])
            # Extra data in 2 has a np array with the displacement in (x, y, theta_around_z)
            dataset[primitive].append(Datapoint(x = inputs, y = env_data.info['extra_data'][i]['displacement'][2]))

        # print('shape x:', dataset[0][0].x.shape)
        # print('shape y:', dataset[0][0].y.shape)
        for i in range(self.nr_primitives):
            self.dynamics_models[i].load_dataset(dataset[i])

    def predict(self, state, action):
        # Assuming that state is the pushing distance and action the index of
        # the action
        inputs = self._clutter2array(state)
        primitive = int(np.floor(action / self.nr_substates))
        if primitive == 1:
            return np.array([0, 0, 0])
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

    def filter_datapoint(datapoint, epsilon=1e-8, filter=0.9, outliers_cutoff=3.8, plot=False):
        '''Assume datapoint is output of _clutter2array'''
        
        import matplotlib.pyplot as plt
        pos_measurements = datapoint[0]
        force_measurements = datapoint[1]

        # Calculate force direction
        # -------------------------
        f = force_measurements[:, :2].copy()
        f = np.nan_to_num(f / norm(f, axis=1).reshape(-1, 1))
        f_norm = np.linalg.norm(f, axis=1)

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
            return np.zeros(2), 0.0

        f = f[start_contact:end_contact, :]

        if plot:
            fig, axs = plt.subplots(2,2)
            axs[0][0].plot(f)
            plt.title('Force')
            axs[0][1].plot(np.linalg.norm(f, axis=1))
            plt.title('norm')

        f[:,0] = filter_signal(signal=f[:,0], filter=filter, outliers_cutoff=outliers_cutoff)
        f[:,1] = filter_signal(signal=f[:,1], filter=filter, outliers_cutoff=outliers_cutoff)
        f = np.nan_to_num(f / norm(f, axis=1).reshape(-1, 1))

        if plot:
            axs[1][0].plot(f)
            plt.title('Filtered force')
            axs[1][1].plot(np.linalg.norm(f, axis=1))
            plt.title('norm')
            plt.show()

        # Velocity direction
        p = pos_measurements[start_contact:end_contact, :2].copy()
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

        # concat

        return signal
