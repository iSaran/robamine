"""
Spit Dynamics Model
===================
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import Agent
from robamine.algo.util import Datapoint, Dataset
from robamine.utils.memory import ReplayBuffer
from collections import deque
from random import Random, shuffle
import numpy as np
import pickle
import math
import os

import logging
logger = logging.getLogger('robamine.algo.splitdynamicsmodel')

net_type = ['feature', 'pose']

class FullyConnectedPrimitiveNetwork(nn.Module):
    def __init__(self, inputs, hidden_units, outputs):
        super(FullyConnectedPrimitiveNetwork, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(inputs, hidden_units[0]))
        i = 0
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            self.hidden_layers[i].weight.data.uniform_(-0.003, 0.003)
            self.hidden_layers[i].bias.data.uniform_(-0.003, 0.003)

        self.out = nn.Linear(hidden_units[i], outputs)

        self.out.weight.data.uniform_(-0.003, 0.003)
        self.out.bias.data.uniform_(-0.003, 0.003)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = nn.functional.relu(x)
        prediction = self.out(x)
        return prediction

class FeaturePredictiveNetwork(FullyConnectedPrimitiveNetwork):
    '''Primitive network taking s_t and predicting (s_t, target_diff_pose)'''
    def __init__(self, feature_dim, hidden_units):
        super(FeaturePredictiveNetwork, self).__init__(inputs=feature_dim, \
            hidden_units=hidden_units, outputs = feature_dim)

class DisplacementPredictiveNetwork(FullyConnectedPrimitiveNetwork):
    '''Primitive network taking s_t and predicting (s_t, target_diff_pose)'''
    def __init__(self, state_dim, hidden_units):
        super(DisplacementPredictiveNetwork, self).__init__(inputs=2, \
            hidden_units=hidden_units, outputs = 3)

class SplitDynamicsModel(Agent):
    def __init__(self,  params):
        super(SplitDynamicsModel, self).__init__(name='SplitDynamicsModel', params=params)
        state_dim, action_dim = params['state_dim'], params['action_dim']

        # The number of networks is the number of high level actions (e.g. push
        # target, push obstacles, grasp). One network per high level action.
        self.nr_primitives = int(len(self.params['feature_hidden_units']))

        # Nr of substates is the number of low level actions, which are
        # represented as different states (e.g. rotations of visual features).
        # This is the number of segments that the incoming states will be
        # splitted to.
        self.nr_substates = int(action_dim / self.nr_primitives)
        self.substate_dim = int(state_dim / self.nr_substates)

        self.device = self.params['device']

        # Create the networks
        self.network = {'feature': nn.ModuleList(), 'pose': nn.ModuleList()}
        for hidden in self.params['feature_hidden_units']:
            self.network['feature'].append(FeaturePredictiveNetwork(self.substate_dim, hidden).to(self.device))
        self.network['pose'] = []
        for hidden in self.params['pose_hidden_units']:
            self.network['pose'].append(DisplacementPredictiveNetwork(self.substate_dim, hidden).to(self.device))

        # Create optimizers and loss
        self.optimizer, self.loss = {}, {}
        self.info['train'] = {}
        self.info['test'] = {}
        for type in net_type:
            self.optimizer[type], self.loss[type] = [], []

            for i in range(self.nr_primitives):
                self.optimizer[type].append(optim.Adam(self.network[type][i].parameters(), lr=self.params[type + '_learning_rate'][i]))
                if self.params[type + '_loss'][i] == 'mse':
                    self.loss[type].append(nn.MSELoss())
                elif self.params[type + '_loss'][i] == 'huber':
                    self.loss[type].append(nn.SmoothL1Loss())
                else:
                    raise ValueError('SplitDynamicsModel: Loss should be mse or huber')

                self.info['train']['loss_' + type + '_' + str(i)] = 0.0
                self.info['test']['loss_' + type + '_' + str(i)] = 0.0

        self.learn_step_counter = 0

        self.dataset = {'train': {'feature': [], 'pose': []}, 'test': {'feature': [], 'pose': []}}


        for i in range(self.nr_primitives):
            for type in net_type:
                self.dataset['train'][type].append(Dataset())
                self.dataset['test'][type].append(Dataset())

        self.learn_steps = 0

    def load_dataset(self, env_data):
        """ Transforms the EnvData x to a Dataset """

        dataset = {'feature': [], 'pose': []}
        for i in range(self.nr_primitives):
            dataset['feature'].append(Dataset())
            dataset['pose'].append(Dataset())

        # Create data points for features and poses
        for i in range(len(env_data.transitions)):
            state = np.split(env_data.transitions[i].state, self.nr_substates)
            next_state = np.split(env_data.transitions[i].next_state, self.nr_substates)
            primitive = int(np.floor(env_data.transitions[i].action / self.nr_substates))
            j = int(env_data.transitions[i].action - np.floor(env_data.transitions[i].action / self.nr_substates) * self.nr_substates)

            # Extra data 0, 1 is the pushing distance and the angle of the action in rads
            pose_input = np.array([env_data.info['extra_data'][i][0], env_data.info['extra_data'][i][1]])

            # Extra data in 2 has a np array with the displacement in (x, y, theta_around_z)
            dataset['pose'][primitive].append(Datapoint(x = pose_input, y = env_data.info['extra_data'][i][2]))

            for j in range(self.nr_substates):
                dataset['feature'][primitive].append(Datapoint(x = state[j], y = next_state[j]))

        # Check if there are NaN values in the dataset and remove the data
        # points that contain them
        try:
            dataset['pose'][0].check()
        except:
            x, y = dataset['pose'][0].to_array()
            indexes = np.nonzero(np.isnan(y))
            for i in reversed(indexes[0]):
                del dataset['pose'][0][i]

        for t in ('feature', 'pose'):
            for i in range(self.nr_primitives):
                dataset[t][i].check()

        # Rescale pose dataset
        dataset['pose'][0].rescale(ranges = [-1, 1])
        dataset['pose'][1].rescale(ranges = [-1, 1])

        # Split to train and test datasets
        for type in net_type:
            for i in range(self.nr_primitives):
                self.dataset['train'][type][i], self.dataset['test'][type][i] = dataset[type][i].split(0.8)

    def predict(self, state, action):
        action_primitive = int(np.floor(action / self.nr_substates))
        substate_index = int(action - np.floor(action / self.nr_substates) * self.nr_substates)
        features = np.split(state, self.nr_substates)
        for j in range(0, self.nr_substates):
            s = torch.FloatTensor(features[j]).to(self.device)
            next_feature = self.network['feature'][action_primitive](s).cpu().detach().numpy()

            if j == 0:
                next_state = next_feature
            else:
                next_state = np.concatenate((next_state, next_feature))

        next_pose = self.network['pose'][action_primitive](s).cpu().detach().numpy()
        return next_state, next_pose

    def learn(self):
        self.learn_steps += 1
        for typ in net_type:
            for i in range(self.nr_primitives):
                if self.learn_steps < self.params[typ + '_nr_epochs'][i]:

                    # Minimbatch update of network
                    minibatches = self.dataset['train'][typ][i].to_minibatches(self.params[typ + '_batch_size'][i])
                    for minibatch in minibatches:
                        batch_x, batch_y = minibatch.to_array()

                        real_x = torch.FloatTensor(batch_x).to(self.device)
                        prediction = self.network[typ][i](real_x)
                        real_y = torch.FloatTensor(batch_y).to(self.device)
                        loss = self.loss[typ][i](prediction, real_y)
                        self.optimizer[typ][i].zero_grad()
                        loss.backward()
                        self.optimizer[typ][i].step()

                    # Calculate loss in train dataset
                    train_x, train_y = self.dataset['train'][typ][i].to_array()
                    real_x = torch.FloatTensor(train_x).to(self.device)
                    prediction = self.network[typ][i](real_x)
                    real_y = torch.FloatTensor(train_y).to(self.device)
                    loss = self.loss[typ][i](prediction, real_y)
                    self.info['train']['loss_' + typ + '_' + str(i)] = loss.detach().cpu().numpy().copy()

                    # Calculate loss in test dataset
                    test_x, test_y = self.dataset['test'][typ][i].to_array()
                    real_x = torch.FloatTensor(test_x).to(self.device)
                    prediction = self.network[typ][i](real_x)
                    real_y = torch.FloatTensor(test_y).to(self.device)
                    loss = self.loss[typ][i](prediction, real_y)
                    self.info['test']['loss_' + typ + '_' + str(i)] = loss.detach().cpu().numpy().copy()

    @classmethod
    def load(cls, file_path):
        model = pickle.load(open(file_path, 'rb'))
        params = model['params']
        self = cls(params)
        self.load_trainable(model['trainable'])
        self.learn_step_counter = model['learn_step_counter']
        logger.info('Agent loaded from %s', file_path)
        return self

    def load_trainable(self, input):
        if isinstance(input, dict):
            trainable = input
            logger.warn('Trainable parameters loaded from dictionary.')
        elif isinstance(input, str):
            fil = pickle.load(open(input, 'rb'))
            trainable = fil['trainable']
            logger.warn('Trainable parameters loaded from: ' + input)
        else:
            raise ValueError('Dict or string is valid')

        for i in range(self.nr_primitives):
            for type in net_type:
                self.network[type][i].load_state_dict(trainable[type][i])

    def save(self, file_path):
        model = {}
        model['params'] = self.params
        model['trainable'] = {'feature': [], 'pose': []}

        for i in range(self.nr_primitives):
            for type in net_type:
                model['trainable'][type].append(self.network[type][i].state_dict())

        model['learn_step_counter'] = self.learn_step_counter
        pickle.dump(model, open(file_path, 'wb'))
