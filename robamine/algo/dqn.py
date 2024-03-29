"""
Deep Q-Network
==============
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import RLAgent
from robamine.utils.memory import ReplayBuffer
from collections import deque
from random import Random
import numpy as np
import pickle
import math
import os

import logging
logger = logging.getLogger('robamine.algo.dqn')

default_params = {
        'name' : 'DQN',
        'replay_buffer_size' : 1e6,
        'batch_size' : 128,
        'discount' : 0.9,
        'epsilon_start' : 0.9,
        'epsilon_end' : 0.05,
        'epsilon_decay' : 0.0,
        'learning_rate' : 1e-4,
        'tau' : 0.999,
        'target_net_updates' : 1000,
        'double_dqn' : True,
        'hidden_units' : [50, 50],
        'device' : 'cuda'
        }

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units):
        super(QNetwork, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(state_dim, hidden_units[0]))
        i = 0
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            self.hidden_layers[i].weight.data.uniform_(-0.003, 0.003)
            self.hidden_layers[i].bias.data.uniform_(-0.003, 0.003)

        self.out = nn.Linear(hidden_units[i], action_dim)

        self.out.weight.data.uniform_(-0.003, 0.003)
        self.out.bias.data.uniform_(-0.003, 0.003)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = nn.functional.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN(RLAgent):
    def __init__(self, state_dim, action_dim, params = default_params):
        super().__init__(state_dim, action_dim, 'DQN', params)

        self.device = self.params['device']

        self.network, self.target_network = QNetwork(state_dim, action_dim, self.params['hidden_units']).to(self.device), QNetwork(state_dim, action_dim, self.params['hidden_units']).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.params['learning_rate'])
        self.loss = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(self.params['replay_buffer_size'])
        self.learn_step_counter = 0

        self.rng = np.random.RandomState()

        self.info['qnet_loss'] = 0
        self.epsilon = self.params['epsilon_start']
        self.info['epsilon'] = self.epsilon

        if self.params['load_buffers'] != '':
            self.replay_buffer = ReplayBuffer.load(os.path.join(self.params['load_buffers']))
            logger.warn("DQN: Preloaded buffer of size " + str(self.replay_buffer.size())+ " from " + self.params['load_buffers'])

    def predict(self, state):
        s = torch.FloatTensor(state).to(self.device)
        action_value = self.network(s).cpu().detach().numpy()
        return np.argmax(action_value)

    def explore(self, state):
        self.epsilon = self.params['epsilon_end'] + \
                       (self.params['epsilon_start'] - self.params['epsilon_end']) * \
                       math.exp(-1 * self.learn_step_counter / self.params['epsilon_decay'])
        if self.rng.uniform(0, 1) >= self.epsilon:
            return self.predict(state)
        return self.rng.randint(0, self.action_dim)

    def learn(self, transition):
        self.info['qnet_loss'] = 0

        self.replay_buffer.store(transition)

        # If we have not enough samples just keep storing transitions to the
        # buffer and thus exit.
        if self.replay_buffer.size() < self.params['batch_size']:
            return

        # Update target network if necessary
        # if self.learn_step_counter > self.params['target_net_updates']:
        #     self.target_network.load_state_dict(self.network.state_dict())
        #     self.learn_step_counter = 0

        new_target_params = {}
        for key in self.target_network.state_dict():
            new_target_params[key] = self.params['tau'] * self.target_network.state_dict()[key] + (1 - self.params['tau']) * self.network.state_dict()[key]
        self.target_network.load_state_dict(new_target_params)

        # Sample from replay buffer
        batch = self.replay_buffer.sample_batch(self.params['batch_size'])
        batch.terminal = np.array(batch.terminal.reshape((batch.terminal.shape[0], 1)))
        batch.reward = np.array(batch.reward.reshape((batch.reward.shape[0], 1)))
        batch.action = np.array(batch.action.reshape((batch.action.shape[0], 1)))

        state = torch.FloatTensor(batch.state).to(self.device)
        action = torch.LongTensor(batch.action.astype(int)).to(self.device)
        next_state = torch.FloatTensor(batch.next_state).to(self.device)
        terminal = torch.FloatTensor(batch.terminal).to(self.device)
        reward = torch.FloatTensor(batch.reward).to(self.device)

        if self.params['double_dqn']:
            best_action = self.network(next_state).max(1)[1]  # action selection
            q_next = self.target_network(next_state).gather(1, best_action.view(self.params['batch_size'], 1))  # action evaluation
        else:
            q_next = self.target_network(next_state).max(1)[0].view(self.params['batch_size'], 1)

        q_target = reward + (1 - terminal) * self.params['discount'] * q_next
        q = self.network(state).gather(1, action)
        loss = self.loss(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.info['qnet_loss'] = loss.detach().cpu().numpy().copy()
        self.info['epsilon'] = self.epsilon

    @classmethod
    def load(cls, file_path):
        model = pickle.load(open(file_path, 'rb'))
        params = model['params']
        self = cls(model['state_dim'], model['action_dim'], params)
        self.load_trainable(model)
        self.learn_step_counter = model['learn_step_counter']
        logger.info('Agent loaded from %s', file_path)
        return self

    def load_trainable(self, input):
        if isinstance(input, dict):
            trainable = input
            logger.warn('Trainable parameters loaded from dictionary.')
        elif isinstance(input, str):
            trainable = pickle.load(open(input, 'rb'))
            logger.warn('Trainable parameters loaded from: ' + input)
        else:
            raise ValueError('Dict or string is valid')

        self.network.load_state_dict(trainable['network'])
        self.target_network.load_state_dict(trainable['target_network'])


    def save(self, file_path):
        model = {}
        model['params'] = self.params
        model['network'] = self.network.state_dict()
        model['target_network'] = self.target_network.state_dict()
        model['learn_step_counter'] = self.learn_step_counter
        model['state_dim'] = self.state_dim
        model['action_dim'] = self.action_dim
        pickle.dump(model, open(file_path, 'wb'))

    def q_value(self, state, action):
        s = torch.FloatTensor(state).to(self.device)
        return self.network(s).cpu().detach().numpy()[action]

    def seed(self, seed):
        self.replay_buffer.seed(seed)
        self.rng.seed(seed)
