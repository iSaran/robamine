"""
The Deep Deterministic Policy Gradient Algorithm
================================================

This module contains the implementation of the Deep Deterministic
Policy Gradient (DDPG) : cite:'lilicarp15'.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import RLAgent
from robamine.utils.memory import ReplayBuffer
from robamine.algo.util import OrnsteinUhlenbeckActionNoise, Transition

import numpy as np


default_params = {
    'name': 'DDPG',
    'replay_buffer_size': 1e6,
    'batch_size': 64,
    'gamma': 0.99,
    'tau': 1e-3,
    'device': 'cuda',
    'actor': {
        'hidden_units': [400, 300],
        'learning_rate': 1e-3,
        'final_layer_init' : [-3e-3, 3e-3]
    },
    'critic': {
        'hidden_units': [400, 300],
        'learning_rate': 1e-3,
        'final_layer_init' : [-3e-3, 3e-3]
    },
    'noise': {
        'name': 'OU',
        'sigma': 0.2
    }
}

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, params=default_params['critic']):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, params['hidden_units'][0])
        self.l2 = nn.Linear(params['hidden_units'][0], params['hidden_units'][1])
        self.l3 = nn.Linear(params['hidden_units'][1], 1)
        nn.init.uniform_(self.l3.weight, params['final_layer_init'][0], params['final_layer_init'][1])
        nn.init.uniform_(self.l3.bias, params['final_layer_init'][0], params['final_layer_init'][1])

    def forward(self, x, u):
        x = nn.functional.relu(self.l1(torch.cat([x, u], x.dim() - 1)))
        x = nn.functional.relu(self.l2(x))
        x = self.l3(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, params=default_params['actor']):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, params['hidden_units'][0])
        self.l2 = nn.Linear(params['hidden_units'][0], params['hidden_units'][1])
        self.l3 = nn.Linear(params['hidden_units'][1], action_dim)
        nn.init.uniform_(self.l3.weight, params['final_layer_init'][0], params['final_layer_init'][1])
        nn.init.uniform_(self.l3.bias, params['final_layer_init'][0], params['final_layer_init'][1])

    def forward(self, x):
        x = nn.functional.relu(self.l1(x))
        x = nn.functional.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x


class DDPG_TORCH(RLAgent):
    def __init__(self, state_dim, action_dim, params=default_params):
        super().__init__(state_dim, action_dim, 'DDPG_TORCH', params)

        self.device = params['device']
        self.params = params

        # Actor
        self.actor = Actor(state_dim, action_dim, self.params['actor']).to(self.device)
        self.target_actor = Actor(state_dim, action_dim, self.params['actor']).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.params['actor']['learning_rate'])

        # Critic
        self.critic = Critic(state_dim, action_dim, self.params['critic']).to(self.device)
        self.target_critic = Critic(state_dim, action_dim, self.params['critic']).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.params['critic']['learning_rate'])

        self.replay_buffer = ReplayBuffer(params['replay_buffer_size'])
        self.exploration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=self.params['noise']['sigma'])

        self.info = {}

    def explore(self, state):
        print(state)
        s = torch.FloatTensor(state).to(self.device)
        noise = self.exploration_noise()
        n = torch.FloatTensor(noise).to(self.device)
        result = self.actor(s) + n
        return result.cpu().detach().numpy()

    def predict(self, state):
        s = torch.FloatTensor(state).to(self.device)
        return self.actor(s).cpu().detach().numpy()

    def learn(self, transition):
        self.replay_buffer.store(transition)

        if self.replay_buffer.count < self.params['batch_size']:
            return

        # sample from replay buffer
        batch = self.replay_buffer.sample_batch(self.params['batch_size'])
        batch.terminal = np.array(batch.terminal.reshape((batch.terminal.shape[0], 1)))
        batch.reward = np.array(batch.reward.reshape((batch.reward.shape[0], 1)))

        state = torch.FloatTensor(batch.state).to(self.device)
        action = torch.FloatTensor(batch.action).to(self.device)
        reward = torch.FloatTensor(batch.reward).to(self.device)
        next_state = torch.FloatTensor(batch.next_state).to(self.device)
        terminal = torch.FloatTensor(batch.terminal).to(self.device)

        # compute the target Q-value
        target_q = self.target_critic(next_state, self.target_actor(next_state))
        target_q = reward + ((1 - terminal) * self.params['gamma'] * target_q).detach()

        # get the current q estimate
        q = self.critic(state, action)

        # critic loss
        critic_loss = nn.functional.mse_loss(q, target_q)
        self.info['critic_loss'] = critic_loss.cpu().detach().numpy().copy()

        # optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.info['actor_loss'] = actor_loss.cpu().detach().numpy().copy()

        # optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft updates for target actor and target critic
        # new_target_actor_params = {}
        # for key in self.target_actor.state_dict():
        #     new_target_actor_params[key] = self.params['tau'] * self.target_actor.state_dict()[key] + \
        #                                    (1 - self.params['tau']) * self.actor.state_dict()[key]
        # self.target_actor.load_state_dict(new_target_actor_params)
        #
        # new_target_critic_params = {}
        # for key in self.target_critic.state_dict():
        #     new_target_critic_params[key] = self.params['tau'] * self.target_critic.state_dict()[key] + \
        #                                    (1 - self.params['tau']) * self.critic.state_dict()[key]
        # self.target_critic.load_state_dict(new_target_critic_params)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

    def q_value(self, state, action):
        s = torch.FloatTensor(state).to(self.device)
        a = torch.FloatTensor(action).to(self.device)
        q = self.critic(s, a).cpu().detach().numpy()
        return q

    def seed(self, seed):
        self.replay_buffer.seed(seed)
        self.exploration_noise.seed(seed)
