"""
Spit Dynamics Model
===================
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import Agent, NetworkModel
from robamine.algo.util import Datapoint, Dataset
import numpy as np
import pickle
import math
import os


from sklearn.decomposition import PCA

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
        self.lstm = nn.LSTM(input_size=inputs, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.hidden2pose_fc = nn.Linear(hidden_dim, outputs)
        self.hidden2pose_fc.weight.data.uniform_(-0.003, 0.003)
        self.hidden2pose_fc.bias.data.uniform_(-0.003, 0.003)

    def forward(self, x):
        hidden = self.init_hidden(x.shape[2])
        out, hidden = self.lstm(x, hidden)
        prediction = self.hidden2pose_fc(out)
        prediction = torch.tanh(prediction)
        return prediction

    def init_hidden(self, x_dim):
        hidden_state = torch.zeros(self.n_layers, x_dim, self.hidden_dim)
        cell_state = torch.zeros(self.n_layers, x_dim, self.hidden_dim)
        hidden = (hidden_state, cell_state)

class AutoEncoderFC(nn.Module):
    def __init__(self, init_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Linear(init_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, init_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.relu(x)
        x = self.decoder(x)
        x = nn.functional.relu(x)
        return x

    def latent(self, x):
        return self.encoder(x)

class FCAutoEncoderModel(NetworkModel):
    def __init__(self, params, inputs, outputs=None, name='AutoEncoderFCModel'):
        self.network = AutoEncoderFC(
            init_dim=inputs,
            latent_dim=params['latent_dim']).to(params['device'])
        super().__init__(params=params, inputs=inputs, outputs=outputs, name=name)

    def load_dataset(self, dataset):
        '''Preprocesses the dataset and loads it to train and test set'''
        assert isinstance(dataset, Dataset)

        x, _ = dataset.to_array()
        dataset = Dataset.from_array(x, x)

        super().load_dataset(dataset)

    def predict(self, state):
        if state.ndim == 1:
            state_ = state.reshape(1, -1)
        else:
            state_ = state

        if self.scaler:
            inputs = self.scaler_x.transform(state_)
        else:
            inputs = state.copy()
        s = torch.FloatTensor(inputs).to(self.device)
        prediction = self.network.latent(s).cpu().detach().numpy()
        return prediction

# Dynamics Models
# ---------------

class FCDynamicsModel(NetworkModel):
    def __init__(self, params, inputs, outputs, name='FCDynamicsModel'):
        ae_params = {
            'batch_size': 64,
            'device': 'cpu',
            'learning_rate': 0.001,
            'loss': 'mse',
            'latent_dim': params['ae_latent_dim']
        }
        # self.autoencoder = FCAutoEncoderModel(ae_params, inputs=inputs)

        self.pca = PCA(n_components=3)

        self.network = FullyConnectedNetwork(
            inputs=32,
            hidden_units=params['hidden_units'],
            outputs=outputs).to(params['device'])

        super().__init__(params=params, inputs=inputs, outputs=outputs, name=name)

    def load_dataset(self, dataset):
        self.autoencoder.load_dataset(dataset)
        for i in range(100):
            self.autoencoder.learn()

        transformed_dataset = dataset
        x, y = transformed_dataset.to_array()
        x = self.autoencoder.predict(x)
        transformed_dataset = Dataset.from_array(x, y)

        super().load_dataset(transformed_dataset)

    def predict(self, state):
        # ndim == 1 is assumed to mean 1 sample (not multiple samples of 1 feature)
        ae_prediction = self.autoencoder.predict(state)
        return super().predict(ae_prediction)


class LSTMDynamicsModel(NetworkModel):
    def __init__(self, params, inputs, outputs, name='LSTMDynamicsModel'):
        self.network = LSTMNetwork(
            inputs=inputs,
            hidden_dim=params['hidden_units'],
            n_layers=params['n_layers'],
            outputs=outputs).to(params['device'])
        super().__init__(params=params, inputs=inputs, outputs=outputs, name=name)

    def load_dataset(self, dataset):
        '''Preprocesses the dataset and loads it to train and test set'''
        assert isinstance(dataset, Dataset)

        # Check and remove NaN values
        try:
            dataset.check()
        except ValueError:
            x, y = dataset.to_array()
            indexes = np.nonzero(np.isnan(y))
            for i in reversed(indexes[0]):
                del dataset[i]

        # Rescale
        if self.scaler:
            data_x, data_y = dataset.to_array()

            init_shape = data_x.shape
            data_x_n = data_x.reshape((init_shape[0] * init_shape[1], init_shape[2]))
            data_x_n = self.scaler_x.fit_transform(data_x_n)
            data_x = data_x_n.reshape(init_shape)

            init_shape = data_y.shape
            data_y_n = data_y.reshape((init_shape[0] * init_shape[1], init_shape[2]))
            data_y_n = self.scaler_y.fit_transform(data_y_n)
            data_y = data_y_n.reshape(init_shape)

            dataset = Dataset.from_array(data_x, data_y)
            dataset.check()

        # Split to train and test datasets
        self.train_dataset, self.test_dataset = dataset.split(0.7)

    def predict(self, state):
        if state.ndim == 1:
            state_ = state.reshape(1, -1)
        else:
            state_ = state

        if self.scaler:
            inputs = self.scaler_x.transform(state_)
        else:
            inputs = state_.copy()
        s = torch.FloatTensor(inputs).to(self.device).unsqueeze(dim=0)
        prediction = self.network(s).cpu().detach().numpy()
        prediction = prediction[:, -1, :].squeeze()  # Obtain the last output
        if self.scaler:
            prediction = self.scaler_y.inverse_transform(prediction.reshape(1, -1)).squeeze()

        return prediction


# Wrappers of Dynamics Models for Split case
# ------------------------------------------

class SplitDynamicsModel(Agent):
    def __init__(self, params, inputs, outputs, model_type, name='SplitDynamicsModel'):
        super().__init__(name=name, params=params)
        state_dim, action_dim = params['state_dim'], params['action_dim']
        self.inputs = inputs
        self.outputs = outputs
        self.model_type = model_type

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

        for i in range(self.nr_primitives):
            params_primitive = {}
            for p in ['hidden_units', 'learning_rate', 'batch_size', 'loss',
                      'n_epochs', 'n_layers', 'ae_latent_dim']:
                if p in self.params:
                    params_primitive[p] = self.params[p][i]
            params_primitive['device'] = self.params['device']

            self.dynamics_models.append(self.model_type(params_primitive, inputs,
                                                        outputs))

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
            self.dynamics_models[i] = self.model_type.load_state_dict(
                state_dict['dynamics_models'][i])

        self.iterations = state_dict['iterations']
        return self

    def seed(self, seed=None):
        super().seed(seed)
        for i in range(self.nr_primitives):
            self.dynamics_models[i].seed(seed)

class FCSplitDynamicsModel(SplitDynamicsModel):
    def __init__(self, params, inputs, outputs, name='FCSplitDynamicsModel'):
        super(FCSplitDynamicsModel, self).__init__(params=params, inputs=inputs, outputs=outputs, model_type=FCDynamicsModel, name=name)

class LSTMSplitDynamicsModel(SplitDynamicsModel):
    def __init__(self, params, inputs, outputs, name='LSTMSplitDynamicsModel'):
        super(LSTMSplitDynamicsModel, self).__init__(params=params, inputs=inputs, outputs=outputs, model_type=LSTMDynamicsModel, name=name)
