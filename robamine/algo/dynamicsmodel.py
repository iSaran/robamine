"""
Spit Dynamics Model
===================
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import Agent
from robamine.algo.util import Datapoint, Dataset
import numpy as np
import pickle
import math
import os

from sklearn.preprocessing import MinMaxScaler

import logging
logger = logging.getLogger('robamine.algo.dynamicsmodel')


class FullyConnectedNetwork(nn.Module):
    def __init__(self, inputs, hidden_units, outputs):
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(inputs, hidden_units[0]))
        i = 0
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1],
                                      hidden_units[i]))
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


class DynamicsModel(Agent):
    def __init__(self, params, inputs, outputs, name='DynamicsModel'):
        super().__init__(name=name, params=params)
        self.inputs = inputs
        self.outputs = outputs
        self.device = self.params['device']

        # Create the networks, optimizers and loss
        self.network = FullyConnectedNetwork(
            inputs=inputs,
            hidden_units=self.params['hidden_units'],
            outputs=outputs).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=self.params['learning_rate'])
        if self.params['loss'] == 'mse':
            self.loss = nn.MSELoss()
        elif self.params['loss'] == 'huber':
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError('DynamicsModel: Loss should be mse or huber')

        # Datasets and their scalers
        self.train_dataset = Dataset()
        self.test_dataset = Dataset()
        self.range = [-1, 1]
        self.min_max_scaler_x = MinMaxScaler(feature_range=self.range)
        self.min_max_scaler_y = MinMaxScaler(feature_range=self.range)

        self.iterations = 0
        self.info['train'] = {'loss': 0.0}
        self.info['test'] = {'loss': 0.0}

    def load_dataset(self, dataset):
        '''Preprocesses the dataset and loads it to train and test set'''

        # Check and remove NaN values
        try:
            dataset.check()
        except ValueError:
            x, y = dataset.to_array()
            indexes = np.nonzero(np.isnan(y))
            for i in reversed(indexes[0]):
                del dataset[i]

        # Rescale
        data_x, data_y = dataset.to_array()
        data_x = self.min_max_scaler_x.fit_transform(data_x)
        data_y = self.min_max_scaler_y.fit_transform(data_y)
        dataset = Dataset.from_array(data_x, data_y)

        dataset.check()

        # Split to train and test datasets
        self.train_dataset, self.test_dataset = dataset.split(0.7)

    def predict(self, state):
        inputs = self.min_max_scaler_x.transform(state.reshape(1, -1))
        s = torch.FloatTensor(inputs).to(self.device)
        prediction = self.network(s).cpu().detach().numpy()
        prediction = self.min_max_scaler_y.inverse_transform(prediction)[0]
        return prediction

    def learn(self):
        '''Run one epoch'''
        self.iterations += 1

        if self.iterations < self.params['nr_epochs']:

            # Minimbatch update of network
            minibatches = self.train_dataset.to_minibatches(
                self.params['batch_size'])
            for minibatch in minibatches:
                batch_x, batch_y = minibatch.to_array()

                real_x = torch.FloatTensor(batch_x).to(self.device)
                prediction = self.network(real_x)
                real_y = torch.FloatTensor(batch_y).to(self.device)
                loss = self.loss(prediction, real_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Calculate loss in train dataset
            train_x, train_y = self.train_dataset.to_array()
            real_x = torch.FloatTensor(train_x).to(self.device)
            prediction = self.network(real_x)
            real_y = torch.FloatTensor(train_y).to(self.device)
            loss = self.loss(prediction, real_y)
            self.info['train']['loss'] = loss.detach().cpu().numpy().copy()

            # Calculate loss in test dataset
            test_x, test_y = self.test_dataset.to_array()
            real_x = torch.FloatTensor(test_x).to(self.device)
            prediction = self.network(real_x)
            real_y = torch.FloatTensor(test_y).to(self.device)
            loss = self.loss(prediction, real_y)
            self.info['test']['loss'] = loss.detach().cpu().numpy().copy()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['trainable'] = self.trainable_dict()
        state_dict['iterations'] = self.iterations
        state_dict['min_max_scaler_x'] = self.min_max_scaler_x
        state_dict['min_max_scaler_y'] = self.min_max_scaler_y
        state_dict['inputs'] = self.inputs
        state_dict['outputs'] = self.outputs
        return state_dict

    def trainable_dict(self):
        return self.network.state_dict()

    def load_trainable_dict(self, trainable):
        self.network.load_state_dict(trainable)

    def load_trainable(self, file_path):
        '''Assume that file path is a pickle with with self.state_dict() '''
        state_dict = pickle.load(open(input, 'rb'))
        self.load_trainable_dict(state_dict['trainable'])

    @classmethod
    def load_state_dict(cls, state_dict):
        self = cls(state_dict['params'], state_dict['inputs'],
                   state_dict['outputs'])
        self.load_trainable_dict(state_dict['trainable'])
        self.iterations = state_dict['iterations']
        self.min_max_scaler_x = state_dict['min_max_scaler_x']
        self.min_max_scaler_y = state_dict['min_max_scaler_y']
        return self
