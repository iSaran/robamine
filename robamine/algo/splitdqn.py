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
import copy

from robamine.algo.util import Transition
import robamine.envs.clutter_utils as clutter
from robamine.utils import cv_tools

import logging
logger = logging.getLogger('robamine.algo.splitdqn')

INFO = True
DEBUG = False
def debug(*args):
    if DEBUG:
        print("DEBUG:splitdqn:" + " ".join(map(str, args)))
def info(*args):
    if INFO:
        print("INFO:splitdqn:" + " ".join(map(str, args)))

default_params = {
        'name' : 'SplitDQN',
        'replay_buffer_size' : [1e6, 1e6],
        'batch_size' : [128, 128],
        'discount' : 0.9,
        'epsilon_start' : 0.9,
        'epsilon_end' : 0.05,
        'epsilon_decay' : 0.0,
        'learning_rate' : [1e-4, 1e-4],
        'tau' : 0.999,
        'target_net_updates' : 1000,
        'double_dqn' : True,
        'hidden_units' : [[50, 50], [50, 50]],
        'loss': ['mse', 'mse'],
        'device' : 'cuda',
        'load_nets' : '',
        'load_buffers' : '',
        'update_iter' : [1, 1, 5]
        }

def split_replay_buffer(buffer, nr_buffers, nr_substates):
    """ Splits a buffer with mixed transitions (from different primitives) to
    one buffer per primitive.
    """
    result = []
    for _ in range(nr_buffers):
        result.append(ReplayBuffer(1e6))
    for i in range(buffer.size()):
        result[int(np.floor(buffer(i).action / nr_substates))].store(buffer(i))
    return result

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

class SplitDQN(RLAgent):
    def __init__(self, state_dim, action_dim, params = default_params):
        super().__init__(state_dim, action_dim, 'SplitDQN', params)

        # The number of networks is the number of high level actions (e.g. push
        # target, push obstacles, grasp). One network per high level action.
        self.nr_network = int(len(self.params['hidden_units']))
        debug('nr_network', self.nr_network)

        # Nr of substates is the number of low level actions, which are
        # represented as different states (e.g. rotations of visual features).
        # This is the number of segments that the incoming states will be
        # splitted to.
        self.nr_substates = int(self.action_dim / self.nr_network)
        debug('nr substates', self.nr_substates)
        self.substate_dim = int(state_dim / self.nr_substates)
        debug('substate dim', self.substate_dim)

        self.device = self.params['device']

        # Create a list of networks and their targets
        self.network, self.target_network = nn.ModuleList(), nn.ModuleList()
        for hidden in self.params['hidden_units']:
            self.network.append(QNetwork(self.substate_dim, 1, hidden).to(self.device))
            debug(self.network)
            self.target_network.append(QNetwork(self.substate_dim, 1, hidden).to(self.device))

        self.optimizer, self.replay_buffer, self.loss = [], [], []
        for i in range(self.nr_network):
            self.optimizer.append(optim.Adam(self.network[i].parameters(), lr=self.params['learning_rate'][i]))
            self.replay_buffer.append(ReplayBuffer(self.params['replay_buffer_size']))
            if self.params['loss'][i] == 'mse':
                self.loss.append(nn.MSELoss())
            elif self.params['loss'][i] == 'huber':
                self.loss.append(nn.SmoothL1Loss())
            else:
                raise ValueError('SplitDQN: Loss should be mse or huber')
            self.info['qnet_' +  str(i) + '_loss'] = 0

        self.learn_step_counter = 0
        self.rng = np.random.RandomState()
        self.epsilon = self.params['epsilon_start']

        if self.params['load_nets'] != '':
            logger.warn("Overwriting the first 2 network params from stored model")
            prev_models = pickle.load(open(os.path.join(self.params['load_nets'], 'model.pkl'), 'rb'))
            for i in range(2):
                self.network[i].load_state_dict(prev_models['network'][i])
                self.target_network[i].load_state_dict(prev_models['target_network'][i])

        if self.params['load_buffers'] != '':
            buffer = ReplayBuffer.load(self.params['load_buffers'])
            self.replay_buffer = split_replay_buffer(buffer, self.nr_network, self.nr_substates)

            for i in range(self.nr_network):
                while (self.replay_buffer[i].size() > self.params['load_buffers_max_size']):
                    self.replay_buffer[i].remove(-1)
                logger.warn("SplitDQN: Preloaded buffer of size " + str(self.replay_buffer[i].size()) + " splitted from " + self.params['load_buffers'])

    def predict(self, state):
        debug('predict: Start')
        action_value = []
        state_ = clutter.get_icra_feature(obs_dict=state, rotations=self.nr_substates)
        state_split = np.split(state_, self.nr_substates)
        for i in range(self.nr_network):
            for j in range(self.nr_substates):
                s = torch.FloatTensor(state_split[j]).to(self.device)
                action_value.append(self.network[i](s).cpu().detach().numpy())
        debug('predict: end')
        return np.argmax(action_value)

    def explore(self, state):
        self.epsilon = self.params['epsilon_end'] + \
                       (self.params['epsilon_start'] - self.params['epsilon_end']) * \
                       math.exp(-1 * self.learn_step_counter / self.params['epsilon_decay'])
        if self.rng.uniform(0, 1) >= self.epsilon:
            return self.predict(state)
        return self.rng.randint(0, self.action_dim)

    def learn(self, transition):
        debug(' ======= SplitDQN Learn start ==========')
        i = int(np.floor(transition.action / self.nr_substates))
        debug('learn step counter:', self.learn_step_counter)
        info('epsilon:', self.epsilon)
        debug('action:', transition.action)
        debug('primitive:', i)
        self.replay_buffer[i].store(self._transitions(transition))

        for _ in range(self.params['update_iter'][i]):
            self.update_net(i)
        debug(' ======= SplitDQN Learn end ==========')


    def update_net(self, i):
        self.info['qnet_' +  str(i) + '_loss'] = 0

        # If we have not enough samples just keep storing transitions to the
        # buffer and thus exit.
        if self.replay_buffer[i].size() < self.params['batch_size'][i]:
            return

        # Update target net's params
        new_target_params = {}
        for key in self.target_network[i].state_dict():
            new_target_params[key] = self.params['tau'] * self.target_network[i].state_dict()[key] + (1 - self.params['tau']) * self.network[i].state_dict()[key]
        self.target_network[i].load_state_dict(new_target_params)

        # Sample from replay buffer
        batch = self.replay_buffer[i].sample_batch(self.params['batch_size'][i])
        batch.terminal = np.array(batch.terminal.reshape((batch.terminal.shape[0], 1)))
        batch.reward = np.array(batch.reward.reshape((batch.reward.shape[0], 1)))

        # Calculate maxQ(s_next, a_next) with max over next actions
        batch_next_state_split = np.split(batch.next_state, self.nr_substates, axis=1)
        q_next = torch.FloatTensor().to(self.device)
        for net in range(self.nr_network):
            for k in range (self.nr_substates):
                next_state = torch.FloatTensor(batch_next_state_split[k]).to(self.device)
                q_next_i = self.target_network[net](next_state)
                q_next = torch.cat((q_next, q_next_i), dim=1)
        q_next = q_next.max(1)[0].view(self.params['batch_size'][i], 1)

        reward = torch.FloatTensor(batch.reward).to(self.device)
        terminal = torch.FloatTensor(batch.terminal).to(self.device)
        q_target = reward + (1 - terminal) * self.params['discount'] * q_next

        # Calculate current q
        split = np.split(batch.state, self.nr_substates, axis=1)
        st = np.zeros((self.params['batch_size'][i], self.substate_dim))
        batch.action = np.subtract(batch.action, i * self.nr_substates)
        for m in range(self.params['batch_size'][i]):
            st[m, :] = split[batch.action[m]][m, :]
        s = torch.FloatTensor(st).to(self.device)
        q = self.network[i](s)

        loss = self.loss[i](q, q_target)
        self.optimizer[i].zero_grad()
        loss.backward()
        self.optimizer[i].step()
        self.info['qnet_' +  str(i) + '_loss'] = loss.detach().cpu().numpy().copy()

        self.learn_step_counter += 1

    def q_value(self, state, action):
        debug('q_value: Start')
        state_ = clutter.get_icra_feature(obs_dict=state, rotations=self.nr_substates)
        split = np.split(state_, self.nr_substates)
        net_index = int(np.floor(action / self.nr_substates))
        substate_index = int(action - np.floor(action / self.nr_substates) * self.nr_substates)
        s = torch.FloatTensor(split[substate_index]).to(self.device)
        debug('q_value: End')
        return self.network[net_index](s).cpu().detach().numpy()

    def seed(self, seed):
        self.replay_buffer.seed(seed)
        self.rng.seed(seed)

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

        for i in range(self.nr_network):
            self.network[i].load_state_dict(trainable['network'][i])
            self.target_network[i].load_state_dict(trainable['target_network'][i])

    def save(self, file_path):
        model = {}
        model['params'] = self.params
        model['network'], model['target_network'] = [], []
        for i in range(self.nr_network):
            model['network'].append(self.network[i].state_dict())
            model['target_network'].append(self.target_network[i].state_dict())
        model['learn_step_counter'] = self.learn_step_counter
        model['state_dim'] = self.state_dim
        model['action_dim'] = self.action_dim
        pickle.dump(model, open(file_path, 'wb'))

    def _transitions(self, transition):
        debug('_transition(): Start')
        # Create rotated states if needed
        debug('State')
        state = clutter.get_icra_feature(obs_dict=transition.state, rotations=self.nr_substates)
        debug('Next State')
        next_state = clutter.get_icra_feature(obs_dict=transition.next_state, rotations=self.nr_substates)
        tran = Transition(state=copy.deepcopy(state),
                          action=transition.action,
                          reward=transition.reward,
                          next_state=copy.deepcopy(next_state),
                          terminal=transition.terminal)
        debug('transitions:tran.state', type(tran.state), tran.state.shape)
        debug('transitions:tran.next_state', type(tran.next_state), tran.next_state.shape)
        debug('transitions:tran.action', tran.action)
        debug('_transition(): end')
        return tran
