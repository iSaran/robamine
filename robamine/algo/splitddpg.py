"""
Split DDPG
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import RLAgent
from robamine.utils.memory import ReplayBuffer
from robamine.algo.util import OrnsteinUhlenbeckActionNoise, Transition

import numpy as np
import pickle


default_params = {
    'name': 'DDPG',
    'replay_buffer_size': 1e6,
    'batch_size': [64, 64],
    'gamma': 0.99,
    'tau': 1e-3,
    'device': 'cpu',
    'nr_primitives': 2,
    'update_iter': [1, 1],
    'actor': {
        'hidden_units': [[400, 300], [400, 300]],
        'learning_rate': 1e-3,
        },
    'critic': {
        'hidden_units': [[400, 300], [400, 300]],
        'learning_rate': 1e-3,
    },
    'noise': {
        'name': 'OU',
        'sigma': 0.2
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
        self.nr_network = self.params['nr_primitives']

        self.device = self.params['device']

        # Create a list of actor-critics and their targets
        self.actor, self.target_actor, self.critic, self.target_critic = nn.ModuleList(), nn.ModuleList(), \
                                                                         nn.ModuleList(), nn.ModuleList()
        for i in range(self.nr_network):
            self.actor.append(Actor(state_dim, action_dim, self.params['actor']['hidden_units'][i]))
            self.target_actor.append(Actor(state_dim, action_dim, self.params['actor']['hidden_units'][i]))
            self.critic.append(Critic(state_dim, action_dim, self.params['critic']['hidden_units'][i]))
            self.target_critic.append(Critic(state_dim, action_dim, self.params['critic']['hidden_units'][i]))

        self.actor_optimizer, self.critic_optimizer, self.replay_buffer = [], [], []
        for i in range(self.nr_network):
            self.critic_optimizer.append(optim.Adam(self.critic.parameters(), self.params['critic']['learning_rate']))
            self.actor_optimizer.append(optim.Adam(self.critic.parameters(), self.params['actor']['learning_rate']))
            self.replay_buffer.append(ReplayBuffer(self.params['replay_buffer_size']))
            self.info['critic_' + str(i) + '_loss'] = 0
            self.info['actor_' + str(i) + '_loss'] = 0

        self.exploration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=self.params['noise']['sigma'])
        self.learn_step_counter = 0

    def predict(self, state):
        s = torch.FloatTensor(state).to(self.device)
        max_q = 0
        for i in range(self.nr_network):
            a = self.actor[i](s)
            q = self.critic[i](s, a).cpu().detach().numpy()
            if q > max_q:
                max_q = q
                max_a = a
        return max_a.cpu().detach().numpy()

    def explore(self, state):
        noise = self.exploration_noise()
        n = torch.FloatTensor(noise).to(self.device)
        s = torch.FloatTensor(self.predict(state))
        a =  s + n
        return a.cpu().detach().numpy()

    def learn(self, transition):
        transition.action = [0, 0, 0.15]
        i = transition.action[0] # first element of action defines the primitive action
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
        action = torch.FloatTensor(batch.action).to(self.device)
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
        self.info['critic_' + str(i) + '_loss'] = critic_loss[i].cpu().detach().numpy().copy()

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
        print('step_counter:', self.learn_step_counter)

    def q_value(self, state, action):
        i = action[0]
        s = torch.FloatTensor(state).to(self.device)
        a = torch.FloatTensor(action[1:]).to(self.device)
        q = self.critic[i](s, a).cpu().detach().numpy()
        return q

    def seed(self, seed):
        self.replay_buffer.seed(seed)
        self.rng.seed(seed)
