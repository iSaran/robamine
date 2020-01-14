"""
Split DDPG
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import RLAgent
from robamine.utils.memory import ReplayBuffer
from robamine.algo.ddppg_torch import Actor, Critic

import numpy as np
import pickle


default_params = {
    'name': 'DDPG',
    'replay_buffer_size': 1e6,
    'batch_size': 64,
    'gamma': 0.99,
    'tau': 1e-3,
    'device': 'cuda',
    'push_target': {
        'actor': {
            'hidden_units': [400, 300],
            'learning_rate': 1e-3,
            'final_layer_init' : [-3e-3, 3e-3]
            },
            'critic': {
            'hidden_units': [400, 300],
            'learning_rate': 1e-3,
            'final_layer_init' : [-3e-3, 3e-3]
        }
    }
        ,

    'noise': {
        'name': 'OU',
        'sigma': 0.2
    }
}


class SplitDDPG(RLAgent):
    def __init__(self, state_dim, action_dim, params=default_params):
        super().__init__(state_dim, action_dim, 'SplitDDG', params)

        # Number of actor-critic networks
        self.nr_network = int(len(self.params['hidden_units']))

        self.device = self.params['device']

        self.actor, self.target_actor, self.critic, self.target_critic = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for hidden in self.params['hidden_units']:
            self.actor.append(Actor(state_dim, action_dim, self.params['actor']))
            self.target_actor.append(Actor(state_dim, action_dim, self.params['actor']))
            self.critic.append(Critic(state_dim, action_dim, self.params['critic']))
            self.target_critic.append(Critic(state_dim, action_dim, self.params['critic']))

        self.actor_optimizer, self.critic_optimizer, self.replay_buffer, \
                        self.actor_loss, self.critic_loss = [], [], [], [], []
        for i in range(self.nr_network):
            self.critic_optimizer.append(optim.Adam(self.critic.parameters(), self.params['critic']['learning_rate']))
            self.actor_optimizer.append(optim.Adam(self.critic.parameters(), self.params['actor']['learning_rate']))
