"""
Split DDPG
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import Agent
from robamine.utils.memory import ReplayBuffer, LargeReplayBuffer, RotatedLargeReplayBuffer

from robamine.envs.clutter_utils import get_rotated_transition, obs_dict2feature, get_observation_dim, get_action_dim

import numpy as np
from math import floor
import pickle

import math
import os

import logging
logger = logging.getLogger('robamine.algo.superviseddqn')

default_params = {
    'name': 'DDPG',
    'replay_buffer_size': 1e6,
    'batch_size': [64, 64],
    'gamma': 0.99,
    'tau': 1e-3,
    'device': 'cpu',
    'actions': [3],
    'update_iter': [1, 1],
    'actor': {
        'hidden_units': [[140, 140], [140, 140]],
        'learning_rate': 1e-3,
        },
    'critic': {
        'hidden_units': [[140, 140], [140, 140]],
        'learning_rate': 1e-3,
    },
    'noise': {
        'name': 'OU',
        'sigma': 0.2
    },
    'epsilon' : {
        'start' : 0.9,
        'end' : 0.05,
        'decay' : 10000,
    }
}

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units):
        super(Critic, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(state_dim + action_dim, hidden_units[0]))
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            stdv = 1. / math.sqrt(self.hidden_layers[i].weight.size(1))
            self.hidden_layers[i].weight.data.uniform_(-stdv, stdv)
            self.hidden_layers[i].bias.data.uniform_(-stdv, stdv)

        self.out = nn.Linear(hidden_units[-1], 1)
        stdv = 1. / math.sqrt(self.out.weight.size(1))
        self.out.weight.data.uniform_(-stdv, stdv)
        self.out.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, u):
        x = torch.cat([x, u], x.dim() - 1)
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        out = self.out(x)
        return out


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units):
        super(Actor, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(state_dim, hidden_units[0]))
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            stdv = 1. / math.sqrt(self.hidden_layers[i].weight.size(1))
            self.hidden_layers[i].weight.data.uniform_(-stdv, stdv)
            self.hidden_layers[i].bias.data.uniform_(-stdv, stdv)

        self.out = nn.Linear(hidden_units[-1], action_dim)
        stdv = 1. / math.sqrt(self.out.weight.size(1))
        self.out.weight.data.uniform_(-stdv, stdv)
        self.out.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        out = torch.tanh(self.out(x))
        return out

    def forward2(self, x):
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = nn.functional.relu(x)

        out = self.out(x)
        return out

class SupervisedActorCritic(Agent):
    def __init__(self, params=default_params):
        super().__init__('SplitDDPG', params)

        self.state_dim = get_observation_dim(params['hardcoded_primitive'])[0]
        self.action_dim = get_action_dim(params['hardcoded_primitive'])[0]

        self.device = self.params['device']
        # Create a list of actor-critics and their targets
        self.actor = Actor(self.state_dim, self.action_dim, self.params['actor']['hidden_units'])
        self.critic = Critic(self.state_dim, self.action_dim, self.params['critic']['hidden_units'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.params['critic']['learning_rate'])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.params['actor']['learning_rate'])
        self.info = {'critic_loss': 0.0, 'actor_loss': 0.0, 'test_critic_loss': 0.0, 'test_actor_loss': 0.0}
        self.replay_buffer = None
        self.n_updates = 0
        self.critic_finished = False
        self.training_indeces = None
        self.test_state = None
        self.test_action = None
        self.test_reward = None

    def predict(self, state):
        output = np.zeros(self.action_dim + 1)
        k = self.params['hardcoded_primitive']
        s = torch.FloatTensor(obs_dict2feature(k, state).array()).to(self.device)
        output[1:] = self.actor(s).detach().cpu().numpy()
        return output

    def get_low_level_action(self, high_level_action):
        # Process a single high level action
        if len(high_level_action.shape) == 1:
            i = int(high_level_action[0])
            return i, high_level_action[1:(self.action_dim + 1)]

        # Process a batch of high levels actions of the same primitive
        elif len(high_level_action.shape) == 2:
            indeces = high_level_action[:, 0].astype('int32')
            assert np.all(indeces == indeces[0])
            i = indeces[0]
            return i, high_level_action[:, 1:(self.action_dim + 1)]

        else:
            raise ValueError(self.name + ': Dimension of a high level action should be 1 or 2.')

    def get_torch_batch(self, indices, rotated=True):
        batch = self.replay_buffer(indices)

        if rotated:
            rotated_batch = []
            for transition in batch:
                random_angle = self.rng.uniform(-180, 180)

                transition.action[0] = self.params['hardcoded_primitive']
                rotated_batch.append(get_rotated_transition(transition, angle=random_angle))
                rotated_batch[-1].action[0] = 0

            reward = torch.FloatTensor([_.reward for _ in rotated_batch]).reshape((len(rotated_batch), 1)).to(
                self.device)
            _, action = self.get_low_level_action(np.array([_.action for _ in rotated_batch]))
            action = torch.FloatTensor(action).to(self.device)
            state = torch.FloatTensor([_.state for _ in rotated_batch]).to(self.device)
        else:
            reward = torch.FloatTensor([_.reward for _ in batch]).reshape((len(batch), 1)).to(
                self.device)
            _, action = self.get_low_level_action(np.array([_.action for _ in batch]))
            action = torch.FloatTensor(action).to(self.device)
            state = torch.FloatTensor([_.state['feature'] for _ in batch]).to(self.device)

        return state, action, reward

    def get_batches(self, indices, batch_size, shuffle=True):
        if shuffle:
            self.rng.shuffle(indices)
        total_size = len(indices)
        batch_size_ = min(batch_size, total_size)
        residual = total_size % batch_size_
        if residual > 0:
            for_splitting = indices[:-residual]
        else:
            for_splitting = indices
        batches = np.split(for_splitting, (total_size - residual) / batch_size_)
        return batches

    def learn(self):

        if not self.critic_finished:
            print('Updating critic. Epoch: ', self.n_updates)
        else:
            print('Updating actor. Epoch: ', self.n_updates)

        batches = self.get_batches(self.training_indeces, self.params['batch_size'], shuffle=True)
        for batch_indices in batches:
            state, action, reward = self.get_torch_batch(batch_indices)

            logging_loss = []
            if not self.critic_finished:
                q = self.critic(state, action)
                critic_loss = nn.functional.mse_loss(q, reward)
                logging_loss.append(critic_loss.detach().cpu().numpy().copy())
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            else:
                state_abs_mean = self.actor.forward2(state).abs().mean()
                preactivation = (state_abs_mean - torch.tensor(1.0)).pow(2)
                if state_abs_mean < torch.tensor(1.0):
                    preactivation = torch.tensor(0.0)
                weight = self.params['actor'].get('preactivation_weight', .05)
                actor_loss = -self.critic(state, self.actor(state)).mean() + weight * preactivation
                logging_loss.append(actor_loss.detach().cpu().numpy().copy())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

        if not self.critic_finished:
            self.info['critic_loss'] = np.mean(logging_loss)
            self.info['test_critic_loss'] = nn.functional.mse_loss(self.critic(self.test_state, self.test_action),
                                                                   self.test_reward).detach().cpu().numpy().copy()
        else:
            self.info['actor_loss'] = np.mean(logging_loss)
            self.info['test_actor_loss'] = (-self.critic(self.test_state, self.actor(
                self.test_state)).mean() + weight * preactivation).detach().cpu().numpy().copy()

        self.n_updates += 1
        if self.n_updates > self.params['critic']['epochs']:
            self.critic_finished = True

    def trainable_dict(self):
        d = {}
        d['critic'] = self.critic.state_dict()
        d['actor'] = self.actor.state_dict()
        return d

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['trainable'] = self.trainable_dict()
        state_dict['state_dim'] = self.state_dim
        state_dict['action_dim'] = self.action_dim
        return state_dict

    def load_trainable_dict(self, trainable):
        self.critic.load_state_dict(trainable['critic'])
        self.actor.load_state_dict(trainable['actor'])

    @classmethod
    def load_state_dict(cls, state_dict):
        params = state_dict['params']
        self = cls(params)
        self.load_trainable(state_dict['trainable'])
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

        self.load_trainable_dict(trainable)

    def load_dataset(self, dataset, perc=0.8):
        self.replay_buffer = RotatedLargeReplayBuffer.load(dataset, mode="r")
        train_scenes = floor(perc * self.replay_buffer.n_scenes)
        self.training_indeces = np.arange(0, train_scenes * self.replay_buffer.rotations, 1)
        test_indeces = np.arange(train_scenes * self.replay_buffer.rotations,
                                      self.replay_buffer.n_scenes * self.replay_buffer.rotations, 1)
        test_batches = self.get_batches(test_indeces, len(test_indeces), shuffle=False)
        self.test_state, self.test_action, self.test_reward = self.get_torch_batch(test_batches[0])

    def q_value(self, state, action):
        i, action_ = self.get_low_level_action(action)
        k = self.params['hardcoded_primitive']
        s = torch.FloatTensor(obs_dict2feature(k, state).array()).to(self.device)
        a = torch.FloatTensor(action_).to(self.device)
        q = self.critic(s, a).detach().cpu().numpy()
        return q
