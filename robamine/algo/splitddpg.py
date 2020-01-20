"""
Split DDPG
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import RLAgent
from robamine.utils.memory import ReplayBuffer
from robamine.algo.util import OrnsteinUhlenbeckActionNoise, NormalNoise, Transition

import numpy as np
import pickle

import math

import logging
logger = logging.getLogger('robamine.algo.splitdqn')

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
            self.hidden_layers[i].weight.data.uniform_(-0.003, 0.003)
            self.hidden_layers[i].bias.data.uniform_(-0.003, 0.003)

        self.out = nn.Linear(hidden_units[-1], 1)
        self.out.weight.data.uniform_(-0.003, 0.003)
        self.out.bias.data.uniform_(-0.003, 0.003)

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
            self.hidden_layers[i].weight.data.uniform_(-0.003, 0.003)
            self.hidden_layers[i].bias.data.uniform_(-0.003, 0.003)

        self.out = nn.Linear(hidden_units[-1], action_dim)
        self.out.weight.data.uniform_(-0.003, 0.003)
        self.out.bias.data.uniform_(-0.003, 0.003)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        out = torch.tanh(self.out(x))
        return out


class SplitDDPG(RLAgent):
    def __init__(self, state_dim, action_dim, params=default_params):
        super().__init__(state_dim, action_dim, 'SplitDDPG', params)

        # The number of networks is the number of primitive actions. One network
        # per primitive action
        self.nr_network = len(self.params['actions'])
        self.actions = self.params['actions']

        self.device = self.params['device']

        # Create a list of actor-critics and their targets
        self.actor, self.target_actor, self.critic, self.target_critic = nn.ModuleList(), nn.ModuleList(), \
                                                                         nn.ModuleList(), nn.ModuleList()
        for i in range(self.nr_network):
            self.actor.append(Actor(state_dim, self.actions[i], self.params['actor']['hidden_units'][i]))
            self.target_actor.append(Actor(state_dim, self.actions[i], self.params['actor']['hidden_units'][i]))
            self.critic.append(Critic(state_dim, self.actions[i], self.params['critic']['hidden_units'][i]))
            self.target_critic.append(Critic(state_dim, self.actions[i], self.params['critic']['hidden_units'][i]))

        self.actor_optimizer, self.critic_optimizer, self.replay_buffer = [], [], []
        for i in range(self.nr_network):
            self.critic_optimizer.append(optim.Adam(self.critic.parameters(), self.params['critic']['learning_rate']))
            self.actor_optimizer.append(optim.Adam(self.actor.parameters(), self.params['actor']['learning_rate']))
            self.replay_buffer.append(ReplayBuffer(self.params['replay_buffer_size']))
            self.info['critic_' + str(i) + '_loss'] = 0
            self.info['actor_' + str(i) + '_loss'] = 0

        self.exploration_noise = []
        for i in range(len(self.actions)):
            if self.params['noise']['name'] == 'OU':
                self.exploration_noise.append(OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.actions[i]), sigma=self.params['noise']['sigma']))
            elif self.params['noise']['name'] == 'Normal':
                self.exploration_noise.append(NormalNoise(mu=np.zeros(self.actions[i]), sigma=self.params['noise']['sigma']))
            else:
                raise ValueError(self.name + ': Exploration noise should be OU or Normal.')
        self.learn_step_counter = 0

    def predict(self, state):
        output = np.zeros(max(self.actions) + 1)
        s = torch.FloatTensor(state).to(self.device)
        max_q = -1e10
        i = 0
        for i in range(self.nr_network):
            a = self.actor[i](s)
            q = self.critic[i](s, a).cpu().detach().numpy()
            if q > max_q:
                max_q = q
                max_a = a.cpu().detach().numpy()
                max_primitive = i

        output[0] = max_primitive
        output[1:(self.actions[max_primitive] + 1)] = max_a
        return output

    def explore(self, state):
        # Calculate epsilon for epsilon-greedy
        start = self.params['epsilon']['end']
        end = self.params['epsilon']['start']
        decay = self.params['epsilon']['decay']
        epsilon =  end + (start - end) * math.exp(-1 * self.learn_step_counter / decay)

        if self.rng.uniform(0, 1) >= epsilon:
            return self.predict(state)
        else:
            i = self.rng.randint(0, len(self.actions))
            s = torch.FloatTensor(state).to(self.device)
            noise = self.exploration_noise[i]()
            n = torch.FloatTensor(noise).to(self.device)
            a = self.actor[i](s) + n
            action = a.cpu().detach().numpy()
            action[action > 1] = 1
            action[action < -1] = -1
            output = np.zeros(max(self.actions) + 1)
            output[0] = i
            output[1:(self.actions[i] + 1)] = action
            return output

    def get_low_level_action(self, high_level_action):
        # Process a single high level action
        if len(high_level_action.shape) == 1:
            i = int(high_level_action[0])
            return i, high_level_action[1:(self.actions[i] + 1)]

        # Process a batch of high levels actions of the same primitive
        elif len(high_level_action.shape) == 2:
            indeces = high_level_action[:, 0].astype('int32')
            assert np.all(indeces == indeces[0])
            i = indeces[0]
            return i, high_level_action[:, 1:(self.actions[i] + 1)]

        else:
            raise ValueError(self.name + ': Dimension of a high level action should be 1 or 2.')

    def learn(self, transition):
        i = int(transition.action[0]) # first element of action defines the primitive action
        self.replay_buffer[i].store(transition)

        for _ in range(self.params['update_iter'][i]):
            self.update_net(i)

    def update_net(self, i):
        self.info['critic_' + str(i) + '_loss'] = 0
        self.info['actor_' + str(i) + '_loss'] = 0

        if self.replay_buffer[i].size() < self.params['batch_size'][i]:
            return

        # Sample from replay buffer
        batch = self.replay_buffer[i].sample_batch(self.params['batch_size'][i])
        batch.terminal = np.array(batch.terminal.reshape((batch.terminal.shape[0], 1)))
        batch.reward = np.array(batch.reward.reshape((batch.reward.shape[0], 1)))

        state = torch.FloatTensor(batch.state).to(self.device)
        _, action_ = self.get_low_level_action(batch.action)
        action = torch.FloatTensor(action_).to(self.device)
        reward = torch.FloatTensor(batch.reward).to(self.device)
        next_state = torch.FloatTensor(batch.next_state).to(self.device)
        terminal = torch.FloatTensor(batch.terminal).to(self.device)

        # Compute the target Q-value
        target_q = self.target_critic[i](next_state, self.target_actor[i](next_state))
        target_q = reward + ((1 - terminal) * self.params['gamma'] * target_q).detach()

        # Get the current q estimate
        q = self.critic[i](state, action)

        # Critic loss
        critic_loss = nn.functional.mse_loss(q, target_q)
        self.info['critic_' + str(i) + '_loss'] = critic_loss.cpu().detach().numpy().copy()

        # Optimize critic
        self.critic_optimizer[i].zero_grad()
        critic_loss.backward()
        self.critic_optimizer[i].step()

        # Compute actor loss
        actor_loss = -self.critic[i](state, self.actor[i](state)).mean()
        self.info['actor_' + str(i) + '_loss'] = actor_loss.cpu().detach().numpy().copy()

        # Optimize actor
        self.actor_optimizer[i].zero_grad()
        actor_loss.backward()
        self.actor_optimizer[i].step()

        # Soft update of target networks
        for param, target_param in zip(self.critic[i].parameters(), self.target_critic[i].parameters()):
            target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

        for param, target_param in zip(self.actor[i].parameters(), self.target_actor[i].parameters()):
            target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

        self.learn_step_counter += 1

    def q_value(self, state, action):
        i, action_ = self.get_low_level_action(action)
        s = torch.FloatTensor(state).to(self.device)
        a = torch.FloatTensor(action_).to(self.device)
        q = self.critic[i](s, a).cpu().detach().numpy()
        return q

    def seed(self, seed):
        self.replay_buffer.seed(seed)
        self.rng.seed(seed)

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['actor'], state_dict['critic'], state_dict['target_actor'], state_dict['target_critic'] = [], [], [], []
        for i in range(self.nr_network):
            state_dict['actor'].append(self.actor[i].state_dict())
            state_dict['critic'].append(self.critic[i].state_dict())
            state_dict['target_actor'].append(self.target_actor[i].state_dict())
            state_dict['target_critic'].append(self.target_critic[i].state_dict())
        state_dict['learn_step_counter'] = self.learn_step_counter
        state_dict['state_dim'] = self.state_dim
        state_dict['action_dim'] = self.action_dim
        return state_dict

    @classmethod
    def load_state_dict(cls, state_dict):
        params = state_dict['params']
        self = cls(state_dict['state_dim'], state_dict['action_dim'], params)
        self.load_trainable(state_dict)
        self.learn_step_counter = state_dict['learn_step_counter']
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
            self.actor[i].load_state_dict(trainable['actor'][i])
            self.critic[i].load_state_dict(trainable['critic'][i])
            self.target_actor[i].load_state_dict(trainable['target_actor'][i])
            self.target_critic[i].load_state_dict(trainable['target_critic'][i])
