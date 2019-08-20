"""
Deep Q-Network with Rotations
=============================
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import Agent
from robamine.algo.dqn import DQN, QNetwork
from robamine.algo.ddpgtorch import ReplayBuffer
from collections import deque
from random import Random
import numpy as np
import pickle
import math

import logging
logger = logging.getLogger('robamine.algo.dqnrot')

default_params = {
        'name' : 'DQNRot',
        'replay_buffer_size' : 1e6,
        'batch_size' : 32,
        'discount' : 0.9,
        'epsilon_start' : 0.9,
        'epsilon_end' : 0.05,
        'epsilon_decay' : 0.0,
        'learning_rate' : 1e-4,
        'tau' : 0.999,
        'target_net_updates' : 1000,
        'double_dqn' : True,
        'hidden_units' : [50, 50],
        'device' : 'cuda',
        'low_level_actions' : 4
        }

class DQNRot(DQN):
    def __init__(self, state_dim, action_dim, params = default_params):
        super(DQN, self).__init__(state_dim, action_dim, 'DQNRot', params)

        self.device = self.params['device']

        # u: nr of high level actions
        # w: nr of low level actions
        # action_dim is u x w
        self.action_dim_w = self.params['low_level_actions']

        assert self.action_dim > self.action_dim_w
        self.action_dim_u = int(self.action_dim / self.action_dim_w)
        assert self.state_dim > self.action_dim_w
        self.true_state_dim = int(self.state_dim / self.action_dim_w)

        self.network = QNetwork(self.true_state_dim, self.action_dim_u, self.params['hidden_units']).to(self.device)
        self.target_network = QNetwork(self.true_state_dim, self.action_dim_u, self.params['hidden_units']).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.params['learning_rate'])
        self.loss = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(self.params['replay_buffer_size'])
        self.learn_step_counter = 0

        self.rng = np.random.RandomState()

        self.info['qnet_loss'] = 0
        self.epsilon = self.params['epsilon_start']
        self.info['epsilon'] = self.epsilon

    def predict(self, state):
        state_split = np.split(state, self.action_dim_w)
        action = []
        for i in range(self.action_dim_u):
            action.append([])
        for j in range(self.action_dim_w):
            s = torch.FloatTensor(state_split[j]).to(self.device)
            actions = self.network(s).cpu().detach().numpy()
            for i in range(self.action_dim_u):
                action[i].append(actions[i])
        action_value = np.array(action).flatten()
        return np.argmax(action_value)

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

        # Calculate maxQ_target(s_next) over all a_next
        batch_next_state_split = np.split(batch.next_state, self.action_dim_w, axis=1)
        q_next = torch.FloatTensor().to(self.device)
        for k in range(self.action_dim_w):
            next_state = torch.FloatTensor(batch_next_state_split[k]).to(self.device)
            q_next_i = self.target_network(next_state)
            q_next = torch.cat((q_next, q_next_i), dim=1)
        max_q_next = q_next.max(1)[0].view(self.params['batch_size'], 1)

        # Calculate the target Q from Bellman
        reward = torch.FloatTensor(batch.reward).to(self.device)
        terminal = torch.FloatTensor(batch.terminal).to(self.device)
        q_target = reward + (1 - terminal) * self.params['discount'] * max_q_next

        # Calculate current q
        split = np.split(batch.state, self.action_dim_w, axis=1)
        st = np.zeros((self.params['batch_size'], self.true_state_dim))
        substate_index = (batch.action - np.floor(batch.action / self.action_dim_w) * self.action_dim_w).astype(int)
        for m in range(self.params['batch_size']):
            st[m, :] = split[substate_index[m]][m, :]
        s = torch.FloatTensor(st).to(self.device)
        net = np.floor(np.array(batch.action.reshape((batch.action.shape[0], 1))) / self.action_dim_w).astype(int)
        action = torch.LongTensor(net).to(self.device)
        q = self.network(s).gather(1, action)

        loss = self.loss(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.info['qnet_loss'] = loss.detach().cpu().numpy().copy()
        self.info['epsilon'] = self.epsilon

    def q_value(self, state, action):
        split = np.split(state, self.action_dim_w)
        substate_index = int(action - np.floor(action / self.action_dim_w) * self.action_dim_w)
        s = torch.FloatTensor(split[substate_index]).to(self.device)
        return self.network(s).cpu().detach().numpy()
