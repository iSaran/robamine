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

# Networks
# --------

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

class LSTMNetwork(nn.Module):
    def __init__(self, inputs, hidden_dim, n_layers, outputs):
        super(LSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(inputs, hidden_dim, n_layers, batch_first=True)
        self.hidden2pose = nn.Linear(hidden_dim, outputs)

    def forward(self, x):
        hidden = self.init_hidden(x.shape[0])
        out, hidden = self.lstm(x, hidden)
        out = out.squeeze()[-1, :]  # Obtain the last output
        prediction = self.hidden2pose(out)
        return prediction

    def init_hidden(self, x_dim):
        hidden_state = torch.zeros(self.n_layers, x_dim, self.hidden_dim)
        cell_state = torch.zeros(self.n_layers, x_dim, self.hidden_dim)
        hidden = (hidden_state, cell_state)


# Dynamics Models
# ---------------

class DynamicsModel(Agent):
    def __init__(self, params, inputs, outputs, name='DynamicsModel'):
        super().__init__(name=name, params=params)
        self.inputs = inputs
        self.outputs = outputs
        self.device = self.params['device']

        # Create the networks, optimizers and loss
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

        if self.iterations < self.params['n_epochs']:

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

class FCDynamicsModel(DynamicsModel):
    def __init__(self, params, inputs, outputs, name='FCDynamicsModel'):
        self.network = FullyConnectedNetwork(
            inputs=inputs,
            hidden_units=params['hidden_units'],
            outputs=outputs).to(params['device'])
        super().__init__(params=params, inputs=inputs, outputs=outputs, name=name)

class LSTMDynamicsModel(DynamicsModel):
    def __init__(self, params, inputs, outputs, name='LSTMDynamicsModel'):
        self.network = LSTMNetwork(
            inputs=inputs,
            hidden_dim=params['hidden_units'],
            n_layers=params['n_layers'],
            outputs=outputs).to(params['device'])
        super().__init__(params=params, inputs=inputs, outputs=outputs, name=name)

# Wrappers of Dynamics Models for Split case
# ------------------------------------------

class SplitDynamicsModel(Agent):
    def __init__(self, params, inputs, outputs, model_type, name='SplitDynamicsModel'):
        super().__init__(name=name, params=params)
        state_dim, action_dim = params['state_dim'], params['action_dim']
        self.inputs = inputs
        self.outputs = outputs

        # The number of networks is the number of high level actions (e.g. push
        # target, push obstacles, grasp). One network per high level action.
        self.nr_primitives = int(len(self.params['hidden_units']))

        # Nr of substates is the number of low level actions, which are
        # represented as different states (e.g. rotations of visual features).
        # This is the number of segments that the incoming states will be
        # splitted to.
        self.nr_substates = int(action_dim / self.nr_primitives)
        self.substate_dim = int(state_dim / self.nr_substates)

        # Storing primitive dynamics models
        self.dynamics_models = []

        # Split the hyperparams and create the primitive dynamics model
        self.info['train'] = {}
        self.info['test'] = {}
        self.train_dataset = []
        self.test_dataset = []

        for i in range(self.nr_primitives):
            params_primitive = {}
            for p in ['hidden_units', 'learning_rate', 'batch_size', 'loss',
                      'n_epochs', 'n_layers']:
                if p in self.params:
                    params_primitive[p] = self.params[p][i]
            params_primitive['device'] = self.params['device']

            self.dynamics_models.append(model_type(params_primitive, inputs,
                                                      outputs))

            self.train_dataset.append(Dataset())
            self.test_dataset.append(Dataset())
            self.info['train']['loss_' + str(i)] = 0.0
            self.info['test']['loss_' + str(i)] = 0.0

        self.iterations = 0

    def predict(self, state, action):
        # Assuming that state is the pushing distance and action the index of
        # the action
        primitive = int(np.floor(action / self.nr_substates))
        prediction = self.dynamics_models[primitive].predict(state)
        return prediction

    def learn(self):
        self.iterations += 1

        for i in range(self.nr_primitives):
            self.dynamics_models[i].learn()
            self.info['train']['loss_' + str(i)] = \
                self.dynamics_models[i].info['train']['loss']
            self.info['test']['loss_' + str(i)] = \
                self.dynamics_models[i].info['test']['loss']

    # Save/load

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['dynamics_models'] = []
        for i in range(self.nr_primitives):
            state_dict['dynamics_models'].append(
                self.dynamics_models[i].state_dict())
        state_dict['iterations'] = self.iterations
        state_dict['inputs'] = self.inputs
        state_dict['outputs'] = self.outputs
        return state_dict

    def load_trainable_dict(self, trainable):
        '''Assume that I have a dict of the form of the dynamics models'''
        for i in range(self.nr_primitives):
            self.dynamics_models[i].load_trainable_dict(
                trainable[i]['trainable'])

    def load_trainable(self, file_path):
        '''Assume that file path is a pickle with with self.state_dict() '''
        state_dict = pickle.load(open(input, 'rb'))
        self.load_trainable_dict(state_dict['dynamics_models'])

    @classmethod
    def load_state_dict(cls, state_dict):
        self = cls(state_dict['params'], state_dict['inputs'],
                   state_dict['outputs'], state_dict['name'])

        for i in range(self.nr_primitives):
            self.dynamics_models[i] = DynamicsModel.load_state_dict(
                state_dict['dynamics_models'][i])

        self.iterations = state_dict['iterations']
        return self

class FCSplitDynamicsModel(SplitDynamicsModel):
    def __init__(self, params, inputs, outputs, name='FCSplitDynamicsModel'):
        super(FCSplitDynamicsModel, self).__init__(params=params, inputs=inputs, outputs=outputs, model_type=FCDynamicsModel, name=name)

class LSTMSplitDynamicsModel(SplitDynamicsModel):
    def __init__(self, params, inputs, outputs, name='LSTMSplitDynamicsModel'):
        super(LSTMSplitDynamicsModel, self).__init__(params=params, inputs=inputs, outputs=outputs, model_type=LSTMDynamicsModel, name=name)
