"""
Split DDPG
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import RLAgent
from robamine.utils.memory import ReplayBuffer
from robamine.algo.util import OrnsteinUhlenbeckActionNoise, NormalNoise, Transition
from robamine.algo.data_compression import AutoEncoder
import robamine.utils.cv_tools as cv_tools

import numpy as np
import pickle

import math
import random

import logging
logger = logging.getLogger('robamine.algo.splitdqn')

default_params = {}

heightmap_params = {
    'layers': 4,
    'encoder': {
        'channels': [32, 64, 128, 256],
        'kernels': [4, 4, 4, 4],
        'strides': [2, 2, 2, 2],
        'padding': [1, 1, 1, 1],
        'pool': [2, 2, 2, 2]
    },
    'decoder': {
        'channels': [256, 128, 64, 32],
        'kernels': [4, 4, 4, 4],
        'stride': [2, 2, 2, 2],
        'padding': [1, 1 ,1 , 1],
        'output_activation': 'linear'
    },
    'device': 'cuda'
}


mask_params = {
    'layers': 4,
    'encoder': {
        'channels': [16, 32, 64, 128],
        'kernels': [4, 4, 4, 4],
        'strides': [2, 2, 2, 2],
        'padding': [1, 1, 1, 1],
        'pool': [2, 2, 2, 2]
    },
    'decoder': {
        'channels': [128, 64, 32, 16],
        'kernels': [4, 4, 4, 4],
        'stride': [2, 2, 2, 2],
        'padding': [1, 1 ,1 , 1],
        'output_activation': 'relu'
    },
    'device': 'cuda'
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
        # u_ = u.clone()
        # if len(u.shape) == 1:
        #     u_[0] = 0
        # else:
        #     u_[:, 0] = torch.zeros((x.shape[0]))
        #
        # x_ = x.clone()
        # angles = np.arange(1, 108, 3)
        # for i in angles:
        #     if len(u.shape) == 1:
        #         x_[i] -= u[0]
        #         if x_[i] < -1.0:
        #             x_[i] = torch.tensor(1.0) - torch.abs(torch.tensor(-1.0) - x_[i])
        #     else:
        #         x_[:, i] -= u[:, 0]
        #         x_[:, i][x_[:, i] < -1.0] = torch.tensor(1.0) - torch.abs(torch.tensor(-1.0) - x_[:, i][x_[:, i] < -1.0])
        #
        # if cout:
        #     print('x:')
        #     print(x.detach().cpu().numpy())
        #     print('---')
        #     print('x_:')
        #     print(x_.detach().cpu().numpy())
        #
        #     print('u:')
        #     print(u.detach().cpu().numpy())
        #     print('----')
        #     print('u_:')
        #     print(u_.detach().cpu().numpy())
        # x = torch.cat([x_, u_], x.dim() - 1)
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

        self.actions_real = []

    def forward(self, x):
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        self.actions_real = self.out(x)
        out = torch.tanh(self.actions_real)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        input_dim = [1, 128, 128]
        latent_dim = 1024

        file_path = '/home/mkiatos/robamine/experiments/2020.02.03_autoencoder/model_mask/model99.pkl'
        with open(file_path, 'rb') as file:
            state_dict = pickle.load(file)
        self.mask_net = AutoEncoder(input_dim, latent_dim, mask_params)
        self.mask_net.load_state_dict(state_dict)

        file_path = '/home/mkiatos/robamine/experiments/2020.02.03_autoencoder/model/model99.pkl'
        with open(file_path, 'rb') as file:
            state_dict = pickle.load(file)
        self.heightmap_net = AutoEncoder(input_dim, latent_dim, heightmap_params)
        self.heightmap_net.load_state_dict(state_dict)

        for params in self.mask_net.parameters():
            params.requires_grad = False
        for params in self.heightmap_net.parameters():
            params.requires_grad = False

    def forward(self, x):
        x1 = torch.unsqueeze(x[:, 0, :, :], 1)
        x2 = torch.unsqueeze(x[:, 1, :, :], 1)
        # cv_tools.plt_plot(x1[0, 0, :, :].detach().cpu().numpy())
        # cv_tools.plt_plot(x2[0, 0, :, :].detach().cpu().numpy())
        z1 = self.heightmap_net.encoder(x1)
        z2 = self.mask_net.encoder(x2)
        # x1_recon = self.heightmap_net.decoder(z1)
        # cv_tools.plt_plot(x1_recon[0, 0, :, :].detach().cpu().numpy())
        # x2_recon = self.mask_net.decoder(z2)
        # cv_tools.plt_plot(x2_recon[0, 0, :, :].detach().cpu().numpy())
        x = torch.cat((z1, z2), 1)
        return x


class SplitDDPG(RLAgent):
    '''
    In info dict it saves for each primitive the loss of the critic and the loss
    of the actor and the qvalues for each primitive. The q values are updated
    only in predict, so during training if you call explore the q values will be
    invalid.
    '''
    def __init__(self, state_dim, action_dim, params=default_params):
        super().__init__(state_dim, action_dim, 'SplitDDPG', params)

        self.obs_dim = 2048

        # The number of networks is the number of primitive actions. One network
        # per primitive action
        self.nr_network = len(self.params['actions'])
        self.actions = self.params['actions']
        self.device = self.params['device']

        # Create a list of actor-critics and their targets
        self.actor, self.target_actor, self.critic, self.target_critic = nn.ModuleList(), nn.ModuleList(), \
                                                                         nn.ModuleList(), nn.ModuleList()
        for i in range(self.nr_network):
            self.actor.append(Actor(self._state_dim(state_dim), self.actions[i],
                                    self.params['actor']['hidden_units'][i]))
            self.target_actor.append(Actor(self._state_dim(state_dim), self.actions[i],
                                           self.params['actor']['hidden_units'][i]))
            self.critic.append(Critic(self._state_dim(state_dim), self.actions[i],
                                      self.params['critic']['hidden_units'][i]))
            self.target_critic.append(Critic(self._state_dim(state_dim), self.actions[i],
                                             self.params['critic']['hidden_units'][i]))

        self.actor_optimizer, self.critic_optimizer, self.replay_buffer = [], [], []
        self.info['q_values'] = []
        for i in range(self.nr_network):
            self.critic_optimizer.append(optim.Adam(self.critic[i].parameters(),
                                                    self.params['critic']['learning_rate']))
            self.actor_optimizer.append(optim.Adam(self.actor[i].parameters(),
                                                   self.params['actor']['learning_rate']))
            self.replay_buffer.append(ReplayBuffer(self.params['replay_buffer_size']))
            self.info['critic_' + str(i) + '_loss'] = 0
            self.info['actor_' + str(i) + '_loss'] = 0
            self.info['preactivation_' + str(i) + '_loss'] = 0
            self.info['q_values'].append(0.0)

        self.exploration_noise = []
        for i in range(len(self.actions)):
            if self.params['noise']['name'] == 'OU':
                self.exploration_noise.append(OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.actions[i]),
                                                                           sigma=self.params['noise']['sigma']))
            elif self.params['noise']['name'] == 'Normal':
                self.exploration_noise.append(NormalNoise(mu=np.zeros(self.actions[i]),
                                                          sigma=self.params['noise']['sigma']))
            else:
                raise ValueError(self.name + ': Exploration noise should be OU or Normal.')
        self.learn_step_counter = 0
        self.preloading_finished = False

        if 'load_actors' in self.params:
            logger.warn("SplitDDPG: Overwriting the actors from the models provided in load_actors param.")
            for i in range(self.nr_network):
                path = self.params['load_actors'][i]
                with open(path, 'rb') as file:
                    pretrained_splitddpg = pickle.load(file)
                    # Assuming that pretrained splitddpg has only one primitive so actor is in 0 index
                    self.actor[i].load_state_dict(pretrained_splitddpg['actor'][0])
                    self.target_actor[i].load_state_dict(pretrained_splitddpg['target_actor'][0])

        self.action_l2 = 0.1
        self.max_action = 5
        self.n_rollout_steps = 30
        self.t_rollout = 0

        # self.replay_buffer[0] = ReplayBuffer.load('/home/mkiatos/Desktop/buffer')
        # print('replay buffer size:', self.replay_buffer[0].size())


    def predict(self, state):
        output = np.zeros(max(self.actions) + 1)
        s = torch.FloatTensor(self._state(state)).to(self.device)
        obs = torch.FloatTensor(self._obs(state)).to(self.device)
        # angle = torch.FloatTensor(self._angle(state)).to(self.device)
        max_q = -1e10

        # for j in range(32):
        #     a = j * 2 / 32
        #     init_d = -1 + a
        #     action = torch.FloatTensor(np.array([init_d, -1, -1]))
        #     q = self.critic[0](s, action, angle)
        #     print(action.detach().cpu().numpy()[:], q.detach().cpu().numpy()[0])
        # print('----')
        # # input('')
        # for j in range(10):
        #     a = j * 2 / 10
        #     init_d = -1 + a
        #     action = torch.FloatTensor(np.array([0, init_d, 0]))
        #     q = self.critic[0](s, action, angle)
        #     print(action.detach().cpu().numpy()[:], q.detach().cpu().numpy()[0])
        # print('----')
        # for j in range(10):
        #     a = j * 2 / 10
        #     init_d = -1 + a
        #     action = torch.FloatTensor(np.array([0, 0, init_d]))
        #     q = self.critic[0](s, action, angle)
        #     print(action.detach().cpu().numpy()[:], q.detach().cpu().numpy()[0])

        # theta = np.linspace(-1, 1, 30)
        # push_d = np.linspace(-1, 1, 30)
        # init_d = np.linspace(-1, 0, 30)
        # for x in theta:
        #     sum_q_value = []
        #     for y in push_d:
        #         for z in init_d:
        #             action = torch.FloatTensor(np.array([z, y, x]))
        #             q = self.critic[0](s, action, angle)
        #             sum_q_value.append(q.detach().cpu().numpy()[0])
        #
        #     print('q_', str(x), ':', np.mean(sum_q_value), np.min(sum_q_value), np.max(sum_q_value))

        for i in range(self.nr_network):
            a = self.actor[i](s)
            q = self.critic[i](s, a).detach().cpu().numpy()
            self.info['q_values'][i] = q[0]
            if q > max_q:
                max_q = q
                max_a = a.detach().cpu().numpy()
                max_primitive = i
            print('actor_output:', a.detach().cpu().numpy()[:])
            print('critic_output:', q[0])

        output[0] = max_primitive
        output[1:(self.actions[max_primitive] + 1)] = max_a
        return output

    def explore(self, state):
        # Calculate epsilon for epsilon-greedy
        start = self.params['epsilon']['start']
        end = self.params['epsilon']['end']
        decay = self.params['epsilon']['decay']
        epsilon =  end + (start - end) * math.exp(-1 * self.learn_step_counter / decay)

        if (self.rng.uniform(0, 1) >= epsilon) and self.preloading_finished:
            pred = self.predict(state)
            i = int(pred[0])
            action = pred[1:self.actions[i] + 1]
            action += self.exploration_noise[i]()
        else:
            i = self.rng.randint(0, len(self.actions))
            action = self.rng.uniform(-1, 1, self.actions[i])
            print('random action')
            print(action)

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
        # i = int(transition.action[0])  # first element of action defines the primitive action
        # if self.t_rollout < self.n_rollout_steps:
        #     self.t_rollout += 1
        #     # store transition
        #     transitions = self._transitions(transition)
        #     for t in transitions:
        #         self.replay_buffer[i].store(t)
        # else:
        #     self.t_rollout = 0
        #
        #     for _ in range(self.params['update_iter'][i]):
        #         self.update_net(i)
        i = int(transition.action[0])  # first element of action defines the primitive action
        transitions = self._transitions(transition)
        for t in transitions:
            self.replay_buffer[i].store(t)

        if not self.preloading_finished:
            if self.replay_buffer[i].size() < 1000:
                print("%d/1000" % self.replay_buffer[i].size())
                return
            else:
                # self.replay_buffer[i].save('/home/mkiatos/Desktop/buffer')
                self.preloading_finished = True

        for _ in range(self.params['update_iter'][i]):
            self.update_net(i)

    def split_state_batch(self, batch_state):
        batch_size = batch_state.shape[0]
        state = np.zeros((batch_size, self._state_dim(self.state_dim)))
        obs = np.zeros((batch_size, self.obs_dim))
        # angle = np.zeros((batch_size, ))
        for i in range(batch_size):
            state[i] = batch_state[i, 0]
            obs[i] = batch_state[i, 1]
            # angle[i] = batch_state[i, 2]
        return state, obs

    def update_net(self, i):
        # Sample from replay buffer
        batch = self.replay_buffer[i].sample_batch(self.params['batch_size'][i])
        batch.terminal = np.array(batch.terminal.reshape((batch.terminal.shape[0], 1)))
        batch.reward = np.array(batch.reward.reshape((batch.reward.shape[0], 1)))

        batch_state, batch_obs = self.split_state_batch(batch.state)
        batch_next_state, batch_next_obs = self.split_state_batch(batch.next_state)

        state = torch.FloatTensor(batch_state).to(self.device)
        obs = torch.FloatTensor(batch_obs).to(self.device)
        _, action_ = self.get_low_level_action(batch.action)
        action = torch.FloatTensor(action_).to(self.device)
        reward = torch.FloatTensor(batch.reward).to(self.device)
        next_state = torch.FloatTensor(batch_next_state).to(self.device)
        next_obs = torch.FloatTensor(batch_next_obs).to(self.device)
        terminal = torch.FloatTensor(batch.terminal).to(self.device)

        # angle = torch.FloatTensor(batch_angle).to(self.device)
        # next_angle = torch.FloatTensor(batch_next_angle).to(self.device)

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

        # Optimize actor
        actor_loss = - self.critic[i](state, self.actor[i](state)).mean()
        state_abs_mean = self.actor[i].actions_real.abs().mean()
        preactivation_loss = (state_abs_mean - torch.tensor(1.0)).pow(2)
        if state_abs_mean < torch.tensor(1.0):
            preactivation_loss = torch.tensor(0.0)
        actor_loss += self.action_l2 * preactivation_loss

        self.info['preactivation_' + str(i) + '_loss'] = preactivation_loss.cpu().detach().numpy().copy()
        self.info['actor_' + str(i) + '_loss'] = actor_loss.cpu().detach().numpy().copy()

        print('losses:', critic_loss.cpu().detach().numpy().copy(),
                         actor_loss.cpu().detach().numpy().copy(),
                         preactivation_loss.cpu().detach().numpy().copy())

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
        s = torch.FloatTensor(self._state(state)).to(self.device)
        # angle = torch.FloatTensor(self._angle(state)).to(self.device)
        a = torch.FloatTensor(action_).to(self.device)
        q = self.critic[i](s, a).cpu().detach().numpy()
        return q

    def seed(self, seed):
        self.replay_buffer.seed(seed)
        self.rng.seed(seed)

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['actor'], state_dict['critic'], state_dict['target_actor'], state_dict[
            'target_critic'] = [], [], [], []
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

    def _state_dim(self, state_dim):
        heightmap_rotations = self.params.get('heightmap_rotations', 0)
        if heightmap_rotations > 0:
            state_network = int(state_dim / heightmap_rotations)
        else:
            state_network = state_dim
        return state_network

    def _state(self, state):
        heightmap_rotations = self.params.get('heightmap_rotations', 0)
        if heightmap_rotations > 0:
            state_split = np.split(state[0], heightmap_rotations)
            s = state_split[0]
        else:
            s = state[0]
        return s

    def _obs(self, state):
        heightmap_rotations = self.params.get('heightmap_rotations', 0)
        if heightmap_rotations > 0:
            state_split = np.split(state[1], heightmap_rotations)
            obs = state_split[0]
        else:
            obs = state[1]
        return obs

    def _angle(self, state):
        return [state[2]]

    def _transitions(self, transition):
        transitions = []
        heightmap_rotations = self.params.get('heightmap_rotations', 0)
        if heightmap_rotations > 0:
            state_split = np.split(transition.state[0], heightmap_rotations)
            next_state_split = np.split(transition.next_state[0], heightmap_rotations)

            obs_split = np.split(transition.state[1], heightmap_rotations)
            next_obs_split = np.split(transition.next_state[1], heightmap_rotations)

            for j in range(heightmap_rotations):

                # actions are btn -1, 1. Change the 1st action which is the angle w.r.t. the target:
                act = transition.action.copy()
                act[1] += j * (2 / heightmap_rotations)
                if act[1] > 1:
                    act[1] = -1 + abs(1 - act[1])

                tran = Transition(state=[state_split[j].copy(), obs_split[j].copy()],
                                  action=act.copy(),
                                  reward=transition.reward,
                                  next_state=[next_state_split[j].copy(), next_obs_split[j].copy()],
                                  terminal=transition.terminal)
                transitions.append(tran)
        else:
            transitions.append(transition)

        return transitions
