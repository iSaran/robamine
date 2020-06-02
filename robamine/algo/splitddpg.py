"""
Split DDPG
"""

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import RLAgent
from robamine.utils.memory import ReplayBuffer, get_batch_indices, LargeReplayBuffer
from robamine.utils.math import min_max_scale
from robamine.utils.orientation import rot_z, Quaternion, transform_poses
from robamine.algo.util import OrnsteinUhlenbeckActionNoise, NormalNoise, Transition
from robamine.envs.clutter_utils import get_observation_dim, get_action_dim, obs_dict2feature, get_table_point_cloud
from robamine.clutter.real_mdp import RealState

import numpy as np
import pickle

import math
from math import pi
import os
import copy

import matplotlib.pyplot as plt
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
    def __init__(self, state_dim, action_dim, hidden_units, rotation_invariant=False):
        super(Critic, self).__init__()
        self.rotation_invariant = rotation_invariant

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
        # Clone an reshape in case of single input in order to have a "batch" shape
        u_ = u.clone()
        x_ = x.clone()

        # Rotation invariant features assumes spherical coordinates
        if self.rotation_invariant:
            if len(u.shape) == 1:
                u_ = u_.reshape((1, -1))
            if len(x_.shape) == 1:
                x_ = x_.reshape((1, -1))

            batch_size = x_.shape[0]
            x__ = x_[:, :120].reshape((batch_size, -1, 4, 3)).clone()
            # Find number of obstacles in order to NOT rotate the zero padding
            n_obstacles = 0
            for i in range(1, x__.shape[1]):
                if not (x__[:, i] == -1).all():
                    n_obstacles += 1
            # Rotate and threshold
            x__[:, :n_obstacles + 1, :, 1] -= u_[:, 0]
            x__[:, :n_obstacles + 1, :, 1][x__[:, :n_obstacles + 1, :, 1] < -1.0] = torch.tensor(1.0) - torch.abs(
                torch.tensor(-1.0) - x__[:, :n_obstacles + 1, :, 1][x__[:, :n_obstacles + 1, :, 1] < -1.0])
            x__[:, :n_obstacles + 1, :, 1][x__[:, :n_obstacles + 1, :, 1] > 1.0] = torch.tensor(-1.0) + torch.abs(
                torch.tensor(1.0) - x__[:, :n_obstacles + 1, :, 1][x__[:, :n_obstacles + 1, :, 1] > 1.0])
            x__ = x__.reshape((batch_size, -1))
            x_[:, :120] = x__
            u_[:, 0] = torch.zeros(u_.shape[0])

        x = torch.cat([x_, u_], x_.dim() - 1)
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
        self.hardcoded_primitive = params['env_params']['hardcoded_primitive']
        self.real_state = params.get('real_state', False)
        self.state_dim = get_observation_dim(self.hardcoded_primitive, real_state=self.real_state)
        self.action_dim = get_action_dim(self.hardcoded_primitive)
        super().__init__(self.state_dim, self.action_dim, 'SplitDDPG', params)

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
            self.critic.append(Critic(self.state_dim[i], self.action_dim[i], self.params['critic']['hidden_units'][i], rotation_invariant=self.params.get('rotation_invariant', False)))
            self.target_critic.append(Critic(self.state_dim[i], self.action_dim[i], self.params['critic']['hidden_units'][i], rotation_invariant=self.params.get('rotation_invariant', False)))

        self.actor_optimizer, self.critic_optimizer, self.replay_buffer = [], [], []
        self.info['q_values'] = []
        for i in range(self.nr_network):
            self.critic_optimizer.append(optim.Adam(self.critic[i].parameters(), self.params['critic']['learning_rate']))
            self.actor_optimizer.append(optim.Adam(self.actor[i].parameters(), self.params['actor']['learning_rate']))
            # self.replay_buffer.append(LargeReplayBuffer(buffer_size=self.params['replay_buffer_size'],
            #                                             obs_dims={'real_state': [RealState.dim()], 'point_cloud': [density ** 2, 2]},
            #                                             action_dim=self.action_dim[i] + 1,
            #                                             path=os.path.join(self.params['log_dir'], 'buffer.hdf5')))
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

        self.obstacle_avoidance_loss = ObstacleAvoidanceLoss(
            distance_range=self.params['obs_avoid_loss']['distance_range'],
            min_dist_range=self.params['obs_avoid_loss']['min_dist_range'],
            device=self.device)

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
            state_ = self.preprocess_real_state(state, 0)
            real_state = RealState(state_, angle=0, sort=True, normalize=True, spherical=False, range_norm=[-1, 1],
                                   translate_wrt_target=False).array()
            s = torch.FloatTensor(real_state).to(self.device)
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
            batch = get_batch_indices(self.replay_buffer[i].size(), self.params['batch_size'][i])[0]
            batch_ = []
            for j in range(len(batch)):
                batch_.append(self.replay_buffer[i](batch[j]))
            self.update_net(i, batch_)
            self.results['network_iterations'] += 1

    def update_net(self, i, batch):
        self.info['critic_' + str(i) + '_loss'] = 0
        self.info['actor_' + str(i) + '_loss'] = 0

        terminal = torch.zeros((len(batch), 1)).to(self.device)
        reward = torch.zeros((len(batch), 1)).to(self.device)
        action = torch.zeros((len(batch), self.action_dim[i])).to(self.device)
        state_real = torch.zeros((len(batch), RealState.dim())).to(self.device)
        next_state_real = torch.zeros((len(batch), RealState.dim())).to(self.device)
        density = self.params['obs_avoid_loss']['density']
        # state_point_cloud = torch.zeros((len(batch), density ** 2, 2))
        max_init_distance = self.params['env_params']['push']['target_init_distance'][1]
        max_obs_bounding_box = np.max(self.params['env_params']['obstacle']['max_bounding_box'])
        density = self.params['obs_avoid_loss']['density']
        for j in range(len(batch)):
            reward[j] = batch[j].reward
            terminal[j] = batch[j].terminal
            _, action_ = self.get_low_level_action(batch[j].action)
            action[j] = torch.FloatTensor(action_).to(self.device)

            state_real[j] = torch.FloatTensor(RealState(batch[j].state, angle=0, sort=True, normalize=True, spherical=False, range_norm=[-1, 1],
                                   translate_wrt_target=False).array()).to(self.device)
            if not terminal[j]:
                next_state_real[j] = torch.FloatTensor(RealState(batch[j].next_state, angle=0, sort=True, normalize=True, spherical=False, range_norm=[-1, 1],
                                                            translate_wrt_target=False).array()).to(self.device)
            # point_cloud_ = get_table_point_cloud(batch[j].state['object_poses'][batch[j].state['object_above_table']],
            #                                      batch[j].state['object_bounding_box'][batch[j].state['object_above_table']],
            #                                      workspace=[max_init_distance, max_init_distance],
            #                                      density=density)
            # state_point_cloud[j, :point_cloud_.shape[0], :] = torch.FloatTensor(point_cloud_).to(self.device)

        # state_real = torch.FloatTensor(batch.state['real_state']).to(self.device)
        # state_point_cloud = torch.FloatTensor(batch.state['point_cloud']).to(self.device)
        # next_state_real = torch.FloatTensor(batch.next_state['real_state']).to(self.device)

        # Compute the target Q-value
        target_q = self.target_critic[i](next_state_real, self.target_actor[i](next_state_real))
        target_q = reward + ((1 - terminal) * self.params['gamma'] * target_q).detach()

        # Get the current q estimate
        q = self.critic[i](state_real, action)

        # Critic loss
        critic_loss = nn.functional.mse_loss(q, target_q)
        self.info['critic_' + str(i) + '_loss'] = float(critic_loss.detach().cpu().numpy())

        # Optimize critic
        self.critic_optimizer[i].zero_grad()
        critic_loss.backward()
        self.critic_optimizer[i].step()

        # Compute preactivation
        state_abs_mean = self.actor[i].forward2(state_real).abs().mean()
        preactivation = (state_abs_mean - torch.tensor(1.0)).pow(2)
        if state_abs_mean < torch.tensor(1.0):
            preactivation = torch.tensor(0.0)
        weight = self.params['actor'].get('preactivation_weight', .05)
        preactivation = weight * preactivation

        actor_action = self.actor[i](state_real)

        critic_loss = - self.critic[i](state_real, actor_action).mean()

        # obs_avoidance = self.obstacle_avoidance_loss(state_point_cloud, actor_action)

        # actor_loss = obs_avoidance + critic_loss
        actor_loss = critic_loss

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
        state_ = self.preprocess_real_state(state, 0)
        real_state = RealState(state_, angle=0, sort=True, normalize=True, spherical=False, range_norm=[-1, 1],
                               translate_wrt_target=False).array()
        s = torch.FloatTensor(real_state).to(self.device)
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
        heightmap_rotations = self.params.get('heightmap_rotations', 1)

        what_i_need = ['object_poses', 'object_bounding_box', 'object_above_table', 'surface_size', 'surface_angle',
                       'max_n_objects']

        max_init_distance = self.params['env_params']['push']['target_init_distance'][1]
        max_obs_bounding_box = np.max(self.params['env_params']['obstacle']['max_bounding_box'])
        density = self.params['obs_avoid_loss']['density']

        # Create rotated states if needed
        angle = np.linspace(0, 2 * np.pi, heightmap_rotations, endpoint=False)
        for j in range(heightmap_rotations):
            state = self.preprocess_real_state(transition.state, angle[j])

            # Real state:
            # real_state = RealState(state, angle=0, sort=True, normalize=True, spherical=False, range_norm=[-1, 1],
            #                        translate_wrt_target=False).array()
            # # Point cloud:
            # point_cloud = max_init_distance * np.ones((density ** 2, 2))
            # point_cloud_ = get_table_point_cloud(state['object_poses'][state['object_above_table']],
            #                                      state['object_bounding_box'][state['object_above_table']],
            #                                      workspace=[max_init_distance, max_init_distance],
            #                                      density=density)
            # point_cloud[:point_cloud_.shape[0]] = point_cloud_

            # Rotate next state
            # real_state_next = np.zeros(RealState.dim())
            # point_cloud_next = max_init_distance * np.ones((density ** 2, 2))
            next_state = self.preprocess_real_state(transition.next_state, angle[j])
            # if not transition.terminal:
            #     next_state = self.preprocess_real_state(transition.next_state, angle[j])
                # Real state:
                # real_state_next = RealState(next_state, angle=0, sort=True, normalize=True, spherical=False, range_norm=[-1, 1],
                #                        translate_wrt_target=False).array()
                # # Point cloud:
                # point_cloud_next = max_init_distance * np.ones((density ** 2, 2))
                # point_cloud_next_ = get_table_point_cloud(next_state['object_poses'][next_state['object_above_table']],
                #                                      next_state['object_bounding_box'][next_state['object_above_table']],
                #                                      workspace=[max_init_distance, max_init_distance],
                #                                      density=density)
                # point_cloud_next[:point_cloud_next_.shape[0]] = point_cloud_next_

            # Rotate action
            # actions are btn -1, 1. Change the 1st action which is the angle w.r.t. the target:
            angle_pi = angle[j]
            if angle[j] > np.pi:
                angle_pi -= 2 * np.pi
            angle_pi = min_max_scale(angle_pi, range=[-np.pi, np.pi], target_range=[-1, 1])
            act = transition.action.copy()
            act[1] += angle_pi
            if act[1] > 1:
                act[1] = -1 + abs(1 - act[1])
            elif act[1] < -1:
                act[1] = 1 - abs(-1 - act[1])

            # statee = {}
            # statee['real_state'] = copy.deepcopy(real_state)
            # statee['point_cloud'] = copy.deepcopy(point_cloud)
            # next_statee = {}
            # next_statee['real_state'] = copy.deepcopy(real_state_next)
            # next_statee['point_cloud'] = copy.deepcopy(point_cloud_next)
            tran = Transition(state=copy.deepcopy(state),
                              action=act.copy(),
                              reward=transition.reward,
                              next_state=copy.deepcopy(next_state),
                              terminal=transition.terminal)
            transitions.append(tran)
        return transitions

    def preprocess_real_state(self, obs_dict, angle):
        what_i_need = ['object_poses', 'object_bounding_box', 'object_above_table', 'surface_size', 'surface_angle',
                       'max_n_objects']
        max_init_distance = self.params['env_params']['push']['target_init_distance'][1]
        max_obs_bounding_box = np.max(self.params['env_params']['obstacle']['max_bounding_box'])

        state = {}
        for key in what_i_need:
            state[key] = copy.deepcopy(obs_dict[key])

        # Keep closest objects
        poses = state['object_poses'][state['object_above_table']]
        if poses.shape[0] == 0:
            return state

        objects_close_target = np.linalg.norm(poses[:, 0:3] - poses[0, 0:3], axis=1) < (
                max_init_distance + max_obs_bounding_box + 0.01)
        state['object_poses'] = state['object_poses'][state['object_above_table']][objects_close_target]
        state['object_bounding_box'] = state['object_bounding_box'][state['object_above_table']][objects_close_target]
        state['object_above_table'] = state['object_above_table'][state['object_above_table']][objects_close_target]
        # Rotate
        poses = state['object_poses'][state['object_above_table']]
        target_pose = poses[0].copy()
        poses = transform_poses(poses, target_pose)
        rotz = np.zeros(7)
        rotz[3:7] = Quaternion.from_rotation_matrix(rot_z(-angle)).as_vector()
        poses = transform_poses(poses, rotz)
        poses = transform_poses(poses, target_pose, target_inv=True)
        target_pose[3:] = np.zeros(4)
        target_pose[3] = 1
        poses = transform_poses(poses, target_pose)
        state['object_poses'][state['object_above_table']] = poses
        state['surface_angle'] = angle
        return state


class ObstacleAvoidanceLoss(nn.Module):
    def __init__(self, distance_range, min_dist_range=[0.002, 0.1], device='cpu'):
        super(ObstacleAvoidanceLoss, self).__init__()
        self.distance_range = distance_range
        self.min_dist_range = min_dist_range
        self.device = device

    def forward(self, point_clouds, actions):
        # Transform the action to cartesian
        theta = min_max_scale(actions[:, 0], range=[-1, 1], target_range=[-pi, pi], lib='torch', device=self.device)
        distance = min_max_scale(actions[:, 1], range=[-1, 1], target_range=self.distance_range, lib='torch',
                                 device=self.device)
        # TODO: Assumes 2 actions!
        x_y = torch.zeros(actions.shape).to(self.device)
        x_y[:, 0] = distance * torch.cos(theta)
        x_y[:, 1] = distance * torch.sin(theta)
        x_y = x_y.reshape(x_y.shape[0], 1, x_y.shape[1]).repeat((1, point_clouds.shape[1], 1))
        diff = x_y - point_clouds
        min_dist = torch.min(torch.norm(diff, p=2, dim=2), dim=1)[0]
        threshold = torch.nn.Threshold(threshold=- self.min_dist_range[1], value= - self.min_dist_range[1])
        min_dist = - threshold(- min_dist)
        # hard_shrink = torch.nn.Hardshrink(lambd=self.min_dist_range[0])
        # min_dist = hard_shrink(min_dist)
        obstacle_avoidance_signal = - min_max_scale(min_dist, range=self.min_dist_range, target_range=[0.0, 200],
                                                    lib='torch', device=self.device)
        close_center_signal = 0.5 - min_max_scale(distance, range=self.distance_range, target_range=[0, .5], lib='torch',
                                                  device=self.device)
        final_signal = close_center_signal + obstacle_avoidance_signal
        return - final_signal.mean()

    def plot(self, point_cloud, density=64):
        from mpl_toolkits.mplot3d import Axes3D
        x_min = np.min(point_cloud[:, 0])
        x_max = np.max(point_cloud[:, 0])
        y_min = np.min(point_cloud[:, 1])
        y_max = np.max(point_cloud[:, 1])
        point_cloud = torch.FloatTensor(point_cloud).to(self.device)

        x = np.linspace(x_min, x_max, density)
        y = np.linspace(y_min, y_max, density)
        x_, y_ = np.meshgrid(x, y)
        z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                theta = min_max_scale(np.arctan2(y_[i, j], x_[i, j]), range=[-np.pi, np.pi], target_range=[-1, 1])
                distance = min_max_scale(np.sqrt(y_[i, j] ** 2 + x_[i, j] ** 2), range=[0, max(x_max, y_max)],
                                         target_range=[-1, 1])
                z[i, j] = self.forward(point_cloud.reshape((1, point_cloud.shape[0], -1)),
                                       torch.FloatTensor([theta, distance]).reshape(1, -1))

        # Uncomment to print min value
        # ij = np.argwhere(z == np.min(z))
        # print(':', x_[ij[0,0], ij[0,1]], y_[ij[0, 0], ij[0, 1]])
        fig = plt.figure()
        axs = Axes3D(fig)
        mycmap = plt.get_cmap('winter')
        surf1 = axs.plot_surface(x_, y_, z, cmap=mycmap)
        fig.colorbar(surf1, ax=axs, shrink=0.5, aspect=5)


