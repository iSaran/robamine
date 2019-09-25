"""
Deep Deterministic Policy Gradient with PyTorch
===============================================

Most of this is taken from https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from robamine.algo.core import Agent, Transition
from collections import deque
from random import Random
import numpy as np
from robamine.algo.util import OrnsteinUhlenbeckActionNoise
import pickle

import logging
logger = logging.getLogger('robamine.algo.ddpg')

default_params = {
        'name' : 'DDPGTorch',
        'replay_buffer_size' : 1e6,
        'batch_size' : 64,
        'discount' : 0.99,
        'tau' : 1e-3,
        'actor' : {
            'hidden_units' : [400, 300],
            'learning_rate' : 1e-4,
            'final_layer_init' : [-3e-3, 3e-3],
            },
        'critic' : {
            'hidden_units' : [400, 300],
            'learning_rate' : 1e-3,
            'final_layer_init' : [-3e-3, 3e-3]
            },
        'noise' : {
            'name' : 'OU',
            'sigma' : 0.2
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

class ReplayBuffer:
    """
    Implementation of the replay experience buffer. Creates a buffer which uses
    the deque data structure. Here you can store experience transitions (i.e.: state,
    action, next state, reward) and sample mini-batches for training.

    You can  retrieve a transition like this:

    Example of use:

    .. code-block:: python

        replay_buffer = ReplayBuffer(10)
        replay_buffer.store()
        replay_buffer.store([0, 2, 1], [1, 2], -12.9, [2, 2, 1], 0)
        # ... more storing
        transition = replay_buffer(2)


    Parameters
    ----------
    buffer_size : int
        The buffer size
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0
        self.random = Random()

    def __call__(self, index):
        """
        Returns a transition from the buffer.

        Parameters
        ----------
        index : int
            The index number of the desired transition

        Returns
        -------
        tuple
            The transition

        """
        return self.buffer[index]

    def store(self, transition):
        """
        Stores a new transition on the buffer.

        Parameters
        ----------
        state : np.ndarray
            The state of the transition
        action : np.ndarray
            The action of the transition
        reward : np.float32
            The reward of the transition
        next_state : np.ndarray
            The next state of the transition
        terminal : np.float32
            1 if this state is terminal. 0 otherwise.
        """
        if self.count < self.buffer_size:
            self.count += 1
        else:
            self.buffer.popleft()
        self.buffer.append(transition)

    def sample_batch(self, given_batch_size):
        """
        Samples a minibatch from the buffer.

        Parameters
        ----------
        given_batch_size : int
            The size of the mini-batch.

        Returns
        -------
        numpy.array
            The state batch
        numpy.array
            The action batch
        numpy.array
            The reward batch
        numpy.array
            The next state batch
        numpy.array
            The terminal batch
        """
        batch = []

        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch = self.random.sample(self.buffer, batch_size)

        state_batch = np.array([_.state for _ in batch])
        action_batch = np.array([_.action for _ in batch])
        reward_batch = np.array([_.reward for _ in batch])
        next_state_batch = np.array([_.next_state for _ in batch])
        terminal_batch = np.array([_.terminal for _ in batch])

        return Transition(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)

    def clear(self):
        """
        Clears the buffer my removing all elements.
        """
        self.buffer.clear()
        self.count = 0

    def size(self):
        """
        Returns the current size of the buffer.

        Returns
        -------
        int
            The number of existing transitions.
        """
        return self.count

    def seed(self, random_seed):
        self.random.seed(random_seed)

class DDPGTorch(Agent):
    def __init__(self, state_dim, action_dim, params = default_params):
        super().__init__(state_dim, action_dim, 'DDPGTorch', params)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.actor = Actor(state_dim, action_dim, params['actor']).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, params['actor']).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), params['actor']['learning_rate'])

        self.critic = Critic(state_dim, action_dim, params['critic']).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, params['critic']).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), params['critic']['learning_rate'])

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.params['replay_buffer_size'])
        self.exploration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim), sigma = self.params['noise']['sigma'])

    @classmethod
    def load(cls, file_path):
        model = pickle.load(open(file_path, 'rb'))
        self = cls(model['state_dim'], model['action_dim'])
        self.actor.load_state_dict(model['actor'])
        self.critic.load_state_dict(model['critic'])
        logger.info('Agent loaded from %s', file_path)
        return self

    def save(self, file_path):
        actor = self.actor.state_dict()
        critic = self.critic.state_dict()
        model = {}
        model['state_dim'] = self.state_dim
        model['action_dim'] = self.action_dim
        model['actor'] = actor
        model['critic'] = critic
        pickle.dump(model, open(file_path, 'wb'))
        logger.info('Agent saved to %s', file_path)

    def explore(self, state):
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

        # If we have not enough samples just keep storing transitions to the
        # buffer and thus exit.
        if self.replay_buffer.size() < self.params['batch_size']:
            return

        batch = self.replay_buffer.sample_batch(self.params['batch_size'])
        batch.terminal = np.array(batch.terminal.reshape((batch.terminal.shape[0], 1)))
        batch.reward = np.array(batch.reward.reshape((batch.reward.shape[0], 1)))

        state = torch.FloatTensor(batch.state).to(self.device)
        action = torch.FloatTensor(batch.action).to(self.device)
        next_state = torch.FloatTensor(batch.next_state).to(self.device)
        terminal = torch.FloatTensor(batch.terminal).to(self.device)
        reward = torch.FloatTensor(batch.reward).to(self.device)

        # Compute the target Q value
        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + ((1 - terminal) * self.params['discount'] * target_q).detach()

        # Get current Q estimate
        current_q = self.critic(state, action)

        critic_loss = nn.functional.mse_loss(current_q, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

    def q_value(self, state, action):
        s = torch.FloatTensor(state).to(self.device)
        a = torch.FloatTensor(action).to(self.device)
        q = self.critic(s, a).cpu().detach().numpy()
        return q

    def seed(self, seed):
        self.replay_buffer.seed(seed)
        self.exploration_noise.seed(seed)
