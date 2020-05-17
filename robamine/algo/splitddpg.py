"""
Split DDPG
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import RLAgent
from robamine.utils.memory import ReplayBuffer
from robamine.utils.math import min_max_scale
from robamine.algo.util import OrnsteinUhlenbeckActionNoise, NormalNoise, Transition
from robamine.envs.clutter_utils import get_observation_dim, get_action_dim, obs_dict2feature

import numpy as np
import pickle

import math
import os

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


class SplitDDPG(RLAgent):
    '''
    In info dict it saves for each primitive the loss of the critic and the loss
    of the actor and the qvalues for each primitive. The q values are updated
    only in predict, so during training if you call explore the q values will be
    invalid.
    '''
    def __init__(self, state_dim, action_dim, params=default_params):
        self.hardcoded_primitive = params['hardcoded_primitive']
        self.real_state = params.get('real_state', False)
        self.state_dim = get_observation_dim(self.hardcoded_primitive, self.real_state)
        self.action_dim = get_action_dim(self.hardcoded_primitive)
        super().__init__(state_dim, action_dim, 'SplitDDPG', params)

        # The number of networks is the number of primitive actions. One network
        # per primitive action
        self.nr_network = len(self.action_dim)

        self.device = self.params['device']



        # Create a list of actor-critics and their targets
        self.actor, self.target_actor, self.critic, self.target_critic = nn.ModuleList(), nn.ModuleList(), \
                                                                         nn.ModuleList(), nn.ModuleList()
        for i in range(self.nr_network):
            self.actor.append(Actor(self.state_dim[i], self.action_dim[i], self.params['actor']['hidden_units'][i]))
            self.target_actor.append(Actor(self.state_dim[i], self.action_dim[i], self.params['actor']['hidden_units'][i]))
            self.critic.append(Critic(self.state_dim[i], self.action_dim[i], self.params['critic']['hidden_units'][i]))
            self.target_critic.append(Critic(self.state_dim[i], self.action_dim[i], self.params['critic']['hidden_units'][i]))

        self.actor_optimizer, self.critic_optimizer, self.replay_buffer = [], [], []
        self.info['q_values'] = []
        for i in range(self.nr_network):
            self.critic_optimizer.append(optim.Adam(self.critic[i].parameters(), self.params['critic']['learning_rate']))
            self.actor_optimizer.append(optim.Adam(self.actor[i].parameters(), self.params['actor']['learning_rate']))
            self.replay_buffer.append(ReplayBuffer(self.params['replay_buffer_size']))
            self.info['critic_' + str(i) + '_loss'] = 0
            self.info['actor_' + str(i) + '_loss'] = 0
            self.info['q_values'].append(0.0)

        self.exploration_noise = []
        for i in range(len(self.action_dim)):
            if self.params['noise']['name'] == 'OU':
                self.exploration_noise.append(OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim[i]), sigma=self.params['noise']['sigma']))
            elif self.params['noise']['name'] == 'Normal':
                self.exploration_noise.append(NormalNoise(mu=np.zeros(self.action_dim[i]), sigma=self.params['noise']['sigma']))
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

        self.n_preloaded_buffer = self.params['n_preloaded_buffer']
        self.log_dir = self.params.get('log_dir', '/tmp')


        self.results = {
            'epsilon': 0.0,
            'network_iterations': 0,
            'replay_buffer_size': []
        }
        for i in range(self.nr_network):
            self.results['replay_buffer_size'].append(0)

    def predict(self, state):
        output = np.zeros(max(self.action_dim) + 1)
        max_q = -1e10
        i = 0
        for i in range(self.nr_network):
            if self.hardcoded_primitive < 0:
                k = i
            else:
                k = self.hardcoded_primitive
            s = torch.FloatTensor(obs_dict2feature(k, state, real_state=self.real_state).array()).to(self.device)
            a = self.actor[i](s)
            q = self.critic[i](s, a).cpu().detach().numpy()
            self.info['q_values'][i] = q[0]
            if q > max_q:
                max_q = q
                max_a = a.cpu().detach().numpy()
                max_primitive = i

        if self.hardcoded_primitive < 0:
            output[0] = max_primitive
        else:
            output[0] = int(self.hardcoded_primitive)
        output[1:(self.action_dim[max_primitive] + 1)] = max_a
        return output

    def explore(self, state):
        # Calculate epsilon for epsilon-greedy
        start = self.params['epsilon']['start']
        end = self.params['epsilon']['end']
        decay = self.params['epsilon']['decay']
        epsilon =  end + (start - end) * math.exp(-1 * self.learn_step_counter / decay)
        self.results['epsilon'] = epsilon

        if (self.rng.uniform(0, 1) >= epsilon) and self.preloading_finished:
            pred = self.predict(state)
            i = int(pred[0])
            action = pred[1:self.action_dim[i] + 1]
            action += self.exploration_noise[i]()
        else:
            i = self.rng.randint(0, len(self.action_dim))
            action = self.rng.uniform(-1, 1, self.action_dim[i])

        action[action > 1] = 1
        action[action < -1] = -1
        output = np.zeros(max(self.action_dim) + 1)
        output[0] = i
        output[1:(self.action_dim[i] + 1)] = action
        return output

    def get_low_level_action(self, high_level_action):
        # Process a single high level action
        if len(high_level_action.shape) == 1:
            i = int(high_level_action[0])
            return i, high_level_action[1:(self.action_dim[i] + 1)]

        # Process a batch of high levels actions of the same primitive
        elif len(high_level_action.shape) == 2:
            indeces = high_level_action[:, 0].astype('int32')
            assert np.all(indeces == indeces[0])
            i = indeces[0]
            return i, high_level_action[:, 1:(self.action_dim[i] + 1)]

        else:
            raise ValueError(self.name + ': Dimension of a high level action should be 1 or 2.')

    def learn(self, transition):
        if self.hardcoded_primitive < 0:
            i = int(transition.action[0])
        else:
            i = 0
        transitions = self._transitions(transition)
        for t in transitions:
            self.replay_buffer[i].store(t)

        self.results['replay_buffer_size'][i] = self.replay_buffer[i].size()

        if self.replay_buffer[i].size() < self.n_preloaded_buffer[i]:
            return
        else:
            self.preloading_finished = True
            # self.replay_buffer[i].save(os.path.join(self.log_dir, 'split_ddpg_preloaded_buffer_' + str(i)))

        for _ in range(self.params['update_iter'][i]):
            batch = self.replay_buffer[i].sample_batch(self.params['batch_size'][i])
            self.update_net(i, batch)
            self.results['network_iterations'] += 1


    def update_net(self, i, batch):
        self.info['critic_' + str(i) + '_loss'] = 0
        self.info['actor_' + str(i) + '_loss'] = 0

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
        self.info['critic_' + str(i) + '_loss'] = float(critic_loss.detach().cpu().numpy())

        # Optimize critic
        self.critic_optimizer[i].zero_grad()
        critic_loss.backward()
        self.critic_optimizer[i].step()

        # Compute actor loss
        state_abs_mean = self.actor[i].forward2(state).abs().mean()
        preactivation = (state_abs_mean - torch.tensor(1.0)).pow(2)
        if state_abs_mean < torch.tensor(1.0):
            preactivation = torch.tensor(0.0)
        weight = self.params['actor'].get('preactivation_weight', .05)
        actor_loss = -self.critic[i](state, self.actor[i](state)).mean() + weight * preactivation

        self.info['actor_' + str(i) + '_loss'] = float(actor_loss.detach().cpu().numpy())

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
        s = torch.FloatTensor(obs_dict2feature(i, state, real_state=self.real_state).array()).to(self.device)
        a = torch.FloatTensor(action_).to(self.device)
        if self.hardcoded_primitive >= 0:
            i = 0
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

    def _transitions(self, transition):
        transitions = []
        heightmap_rotations = self.params.get('heightmap_rotations', 0)
        if heightmap_rotations > 0:


            for j in range(heightmap_rotations):
                angle = 2 / heightmap_rotations * j

                # TODO this 360 might be different in grasp target
                angle_deg = min_max_scale(angle, [-1, 1], [0, 360])

                state = obs_dict2feature(int(transition.action[0]), transition.state, angle=angle_deg, real_state=self.real_state).array()
                next_state = np.zeros(get_observation_dim(int(transition.action[0]), self.real_state)[0])
                if not transition.terminal:
                    next_state = obs_dict2feature(int(transition.action[0]), transition.next_state, angle=angle_deg, real_state=self.real_state).array()

                # actions are btn -1, 1. Change the 1st action which is the angle w.r.t. the target:
                act = transition.action.copy()
                act[1] += angle
                if act[1] > 1:
                    act[1] = -1 + abs(1 - act[1])

                tran = Transition(state=state.copy(),
                                  action=act.copy(),
                                  reward=transition.reward,
                                  next_state=next_state.copy(),
                                  terminal=transition.terminal)
                transitions.append(tran)
        else:
            state = obs_dict2feature(int(transition.action[0]), transition.state, real_state=self.real_state).array()
            next_state = np.zeros(get_observation_dim(int(transition.action[0]), real_state=self.real_state)[0])
            if not transition.terminal:
                next_state = obs_dict2feature(int(transition.action[0]), transition.next_state, real_state=self.real_state).array()
            tran = Transition(state=state,
                              action=transition.action,
                              reward=transition.reward,
                              next_state=next_state,
                              terminal=transition.terminal)
            transitions.append(tran)

        return transitions
