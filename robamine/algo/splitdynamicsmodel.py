"""
Spit Dynamics Model
===================
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import Agent
from robamine.algo.dynamicsmodel import DynamicsModel
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


class SplitDynamicsModel(Agent):
    def __init__(self, params, inputs, outputs, name='SplitDynamicsModel'):
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
                      'nr_epochs']:
                params_primitive[p] = self.params[p][i]
            params_primitive['device'] = self.params['device']

            self.dynamics_models.append(DynamicsModel(params_primitive, inputs,
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
