from robamine.algo.core import EvalWorld, TrainEvalWorld
from robamine import rb_logging
import yaml
import socket

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.core import RLAgent
from robamine.utils.memory import ReplayBuffer, get_batch_indices
from robamine.utils.math import min_max_scale
from robamine.algo.util import OrnsteinUhlenbeckActionNoise, NormalNoise, Transition
import robamine.envs.clutter_utils as clutter
import h5py

import numpy as np
import pickle

import math
from math import pi
import os
import copy

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger('robamine')

from robamine.experiments.ral.supervised_push_obstacle import Actor, PushObstacleRealPolicyDeterministic, ObsDictPolicy

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, rotation_invariant=False):
        super(Critic, self).__init__()
        self.rotation_invariant = rotation_invariant

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

    def __init__(self, state_dim, action_dim, params):
        self.hardcoded_primitive = params['env_params']['hardcoded_primitive']
        self.real_state = params.get('real_state', False)
        obs_dim_all = [clutter.RealRealState.dim(), clutter.RealRealState.dim()]
        self.state_dim = [obs_dim_all[self.hardcoded_primitive]]
        self.action_dim = clutter.get_action_dim(self.hardcoded_primitive)
        super().__init__(self.state_dim, self.action_dim, 'SplitDDPG', params)
        self.asymmetric = self.params.get('asymmetric', 'asymmetric')

        # Load autoencoder
        if self.asymmetric == 'asymmetric' or self.asymmetric == 'visual':
            import robamine.algo.conv_vae as ae
            with open(self.params['actor']['autoencoder']['model'], 'rb') as file:
                # model = pickle.load(file)
                model = torch.load(file, map_location='cpu')

            latent_dim = model['encoder.fc.weight'].shape[0]
            ae_params = ae.params
            ae_params['device'] = 'cpu'
            self.ae = ae.ConvVae(latent_dim, ae_params)
            self.ae.load_state_dict(model)

            with open(self.params['actor']['autoencoder']['scaler'], 'rb') as file:
                self.scaler = pickle.load(file)

            visual_dim = ae.LATENT_DIM + 4  # TODO: hardcoded the extra dim for surface edges
            if self.asymmetric == 'asymmetric':
                self.actor_state_dim = visual_dim
            elif self.asymmetric == 'visual':
                self.actor_state_dim = visual_dim
                for i in range(len(self.state_dim)):
                    self.state_dim[i] = visual_dim

        if self.asymmetric == 'real':
            self.actor_state_dim = clutter.RealRealState.dim()

        # The number of networks is the number of primitive actions. One network
        # per primitive action
        self.nr_network = len(self.action_dim)

        self.device = self.params['device']

        # Create a list of actor-critics and their targets
        self.actor, self.target_actor, self.critic, self.target_critic = nn.ModuleList(), nn.ModuleList(), \
                                                                         nn.ModuleList(), nn.ModuleList()
        for i in range(self.nr_network):
            self.actor.append(Actor(self.actor_state_dim, self.action_dim[i], self.params['actor']['hidden_units'][i]))
            self.target_actor.append(
                Actor(self.actor_state_dim, self.action_dim[i], self.params['actor']['hidden_units'][i]))
            self.critic.append(Critic(self.state_dim[i], self.action_dim[i], self.params['critic']['hidden_units'][i],
                                      rotation_invariant=self.params.get('rotation_invariant', False)))
            self.target_critic.append(
                Critic(self.state_dim[i], self.action_dim[i], self.params['critic']['hidden_units'][i],
                       rotation_invariant=self.params.get('rotation_invariant', False)))

        self.actor_optimizer, self.critic_optimizer, self.replay_buffer = [], [], []
        self.info['q_values'] = []
        for i in range(self.nr_network):
            self.critic_optimizer.append(
                optim.Adam(self.critic[i].parameters(), self.params['critic']['learning_rate']))
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
        for i in range(self.nr_network):
            if self.params['noise']['name'] == 'OU':
                self.exploration_noise.append(
                    OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim[i]), sigma=self.params['noise']['sigma']))
            elif self.params['noise']['name'] == 'Normal':
                self.exploration_noise.append(
                    NormalNoise(mu=np.zeros(self.action_dim[i]), sigma=self.params['noise']['sigma']))
            else:
                raise ValueError(self.name + ': Exploration noise should be OU or Normal.')
        self.learn_step_counter = 0
        self.preloading_finished = False

        if 'pretrained' in self.params['actor']:
            logger.warn("SplitDDPG: Overwriting the actors from the models provided in load_actors param.")
            for i in range(self.nr_network):
                path = self.params['actor']['pretrained'][i]
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

        self.max_init_distance = self.params['env_params']['push']['target_init_distance'][1]
        if self.params.get('save_heightmaps_disk', False):
            self.file_heightmaps = h5py.File(os.path.join(self.log_dir, 'heightmaps_dataset.h5py'), "a")
            self.file_heightmaps.create_dataset('heightmap_mask', (5000, 2, 386, 386), dtype='f')
            self.file_heightmaps_counter = 0

    def predict(self, state):
        output = np.zeros(max(self.action_dim) + 1)
        max_q = -1e10
        valid_nets = np.arange(0, self.nr_network, 1)
        if self.nr_network > 1 and not clutter.push_obstacle_feature_includes_affordances(state):
            valid_nets = np.delete(valid_nets, [1])
        for i in valid_nets:
            actor_state, critic_state = self._get_actor_critic_state(state,
                                                                     primitive=self.network_to_primitive_index(i))

            s = torch.FloatTensor(critic_state).to(self.device)
            actor_state = torch.FloatTensor(actor_state).to(self.device)
            a = self.actor[i](actor_state)
            q = self.critic[i](s, a).cpu().detach().numpy()
            self.info['q_values'][i] = q[0]
            if q > max_q:
                max_q = q
                max_a = a.cpu().detach().numpy()
                max_primitive = i

        output[0] = self.network_to_primitive_index(max_primitive)
        output[1:(self.action_dim[max_primitive] + 1)] = max_a
        return output

    def explore(self, state):
        return self.exploration_policy(state)
        # if self.hardcoded_primitive >= 0:
        #     return self.exploration_policy(state)
        # else:
        #     return self.exploration_policy_combo(state)

    def exploration_policy(self, state):
        # Calculate epsilon for epsilon-greedy
        start = self.params['epsilon']['start']
        end = self.params['epsilon']['end']
        decay = self.params['epsilon']['decay']
        epsilon = end + (start - end) * math.exp(-1 * self.learn_step_counter / decay)
        self.results['epsilon'] = epsilon

        output = np.zeros(max(self.action_dim) + 1)
        if (self.rng.uniform(0, 1) >= epsilon) and self.preloading_finished:
            pred = self.predict(state)
            i = int(pred[0])
            action = pred[1:self.action_dim[self.primitive_to_network_index(i)] + 1]
            action += self.exploration_noise[self.primitive_to_network_index(i)]()
            action[action > 1] = 1
            action[action < -1] = -1
            output[0] = i
            output[1:(self.action_dim[self.primitive_to_network_index(i)] + 1)] = action
        else:
            valid_nets = np.arange(0, self.nr_network, 1)
            if self.nr_network > 1 and not clutter.push_obstacle_feature_includes_affordances(state):
                valid_nets = np.delete(valid_nets, [1])
            i = valid_nets[self.rng.randint(0, len(valid_nets))]
            action = self.rng.uniform(-1, 1, self.action_dim[i])
            action[action > 1] = 1
            action[action < -1] = -1
            output[0] = self.network_to_primitive_index(i)
            output[1:(self.action_dim[i] + 1)] = action

        return output

    def exploration_policy_combo(self, state):
        # Calculate epsilon for epsilon-greedy
        start = self.params['epsilon']['start']
        end = self.params['epsilon']['end']
        decay = self.params['epsilon']['decay']
        epsilon = end + (start - end) * math.exp(-1 * self.learn_step_counter / decay)
        self.results['epsilon'] = epsilon

        output = np.zeros(max(self.action_dim) + 1)
        if (self.rng.uniform(0, 1) >= epsilon) and self.preloading_finished:
            pred = self.predict(state)
            i = int(pred[0])
            action = pred[1:self.action_dim[self.primitive_to_network_index(i)] + 1]
            action += self.exploration_noise[self.primitive_to_network_index(i)]()
            action[action > 1] = 1
            action[action < -1] = -1
            output[0] = i
            output[1:(self.action_dim[self.primitive_to_network_index(i)] + 1)] = action
        else:
            i = self.rng.randint(0, self.nr_network)
            feature, _ = self._get_actor_critic_state(state, primitive=self.network_to_primitive_index(i))
            actor_state = torch.FloatTensor(feature).to(self.device)
            action = self.actor[i](actor_state).detach().cpu().numpy().copy()
            action += self.exploration_noise[self.primitive_to_network_index(i)]()
            action[action > 1] = 1
            action[action < -1] = -1
            output[0] = self.network_to_primitive_index(i)
            output[1:(self.action_dim[i] + 1)] = action

        return output

    def get_low_level_action(self, high_level_action):
        # Process a single high level action
        if len(high_level_action.shape) == 1:
            i = int(high_level_action[0])
            return i, high_level_action[1:(self.action_dim[self.primitive_to_network_index(i)] + 1)]

        # Process a batch of high levels actions of the same primitive
        elif len(high_level_action.shape) == 2:
            indeces = high_level_action[:, 0].astype('int32')
            assert np.all(indeces == indeces[0])
            i = indeces[0]
            return i, high_level_action[:, 1:(self.action_dim[self.primitive_to_network_index(i)] + 1)]

        else:
            raise ValueError(self.name + ': Dimension of a high level action should be 1 or 2.')

    def learn(self, transition):
        if self.hardcoded_primitive < 0:
            i = int(transition.action[0])
        else:
            i = 0
        transitions = self._transitions(transition)
        h = transition.state['heightmap_mask'][0]
        m = transition.state['heightmap_mask'][1]
        if self.params.get('save_heightmaps_disk', False):
            if self.file_heightmaps_counter < 5000:
                self.file_heightmaps['heightmap_mask'][self.file_heightmaps_counter, :, :, :] = transition.state[
                    'heightmap_mask'].copy()
                self.file_heightmaps_counter += 1

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
        state_real = torch.zeros((len(batch), self.state_dim[i])).to(self.device)
        state_real_actor = torch.zeros((len(batch), self.actor_state_dim)).to(self.device)
        next_state_real_actor = torch.zeros((len(batch), self.actor_state_dim)).to(self.device)
        next_state_real = torch.zeros((len(batch), self.state_dim[i])).to(self.device)
        for j in range(len(batch)):
            reward[j] = batch[j].reward
            terminal[j] = batch[j].terminal
            _, action_ = self.get_low_level_action(batch[j].action)
            action[j] = torch.FloatTensor(action_).to(self.device)

            state_real[j] = torch.FloatTensor(batch[j].state['critic']).to(self.device)
            state_real_actor[j] = torch.FloatTensor(batch[j].state['actor']).to(self.device)

            if not terminal[j]:
                next_state_real[j] = torch.FloatTensor(batch[j].next_state['critic']).to(self.device)
                next_state_real_actor[j] = torch.FloatTensor(batch[j].next_state['actor']).to(self.device)

        # Compute the target Q-value
        target_q = self.target_critic[i](next_state_real, self.target_actor[i](next_state_real_actor))
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

        if self.learn_step_counter > self.params['actor']['start_training_at']:
            # Compute preactivation
            state_abs_mean = self.actor[i].forward2(state_real_actor).abs().mean()
            preactivation = (state_abs_mean - torch.tensor(1.0)).pow(2)
            if state_abs_mean < torch.tensor(1.0):
                preactivation = torch.tensor(0.0)
            weight = self.params['actor'].get('preactivation_weight', .05)
            preactivation = weight * preactivation

            actor_action = self.actor[i](state_real_actor)

            critic_loss = - self.critic[i](state_real, actor_action).mean()

            # obs_avoidance = self.obstacle_avoidance_loss(state_point_cloud, actor_action)

            # actor_loss = obs_avoidance + critic_loss
            actor_loss = critic_loss + preactivation

            self.info['actor_' + str(i) + '_loss'] = float(actor_loss.detach().cpu().numpy())

            # Optimize actor
            self.actor_optimizer[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[i].step()

            for param, target_param in zip(self.actor[i].parameters(), self.target_actor[i].parameters()):
                target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

        for param, target_param in zip(self.critic[i].parameters(), self.target_critic[i].parameters()):
            target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

        self.learn_step_counter += 1

    def q_value(self, state, action):
        i, action_ = self.get_low_level_action(action)
        _, critic_state = self._get_actor_critic_state(state, primitive=self.network_to_primitive_index(i), angle=0)
        s = torch.FloatTensor(critic_state).to(self.device)
        a = torch.FloatTensor(action_).to(self.device)
        if self.hardcoded_primitive >= 0:
            i = 0
        q = self.critic[i](s, a).cpu().detach().numpy()
        return q

    def seed(self, seed):
        for i in range(len(self.replay_buffer)):
            self.replay_buffer[i].seed(seed)
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
        # Uncomment for eval in different pc
        # state_dict['params']['actor']['autoencoder']['model'] = '/home/iason/robamine_logs/2020.01.16.split_ddpg/VAE/model.pkl'
        # state_dict['params']['actor']['autoencoder']['scaler'] = '/home/iason/robamine_logs/2020.01.16.split_ddpg/VAE/normalizer.pkl'
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

        # Create rotated states if needed
        angle = np.linspace(0, 2 * np.pi, heightmap_rotations, endpoint=False)
        state, next_state = {}, {}
        for j in range(heightmap_rotations):
            transition.state['init_distance_from_target'] = transition.next_state['init_distance_from_target']

            state['actor'], state['critic'] = self._get_actor_critic_state(state=transition.state,
                                                                           primitive=transition.action[0],
                                                                           angle=angle[j], plot=False)

            next_state['actor'], next_state['critic'] = self._get_actor_critic_state(state=transition.next_state,
                                                                                     primitive=transition.action[0],
                                                                                     angle=angle[j],
                                                                                     terminal=transition.terminal,
                                                                                     plot=False)

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

    def primitive_to_network_index(self, i):
        if self.hardcoded_primitive < 0:
            return i
        return 0

    def network_to_primitive_index(self, i):
        if self.hardcoded_primitive < 0:
            return i
        return int(self.hardcoded_primitive)

    def _get_actor_critic_state(self, state, primitive, angle=0, terminal=False, plot=False):
        if self.asymmetric == 'asymmetric':
            actor_state = clutter.get_asymmetric_actor_feature_from_dict(state, self.ae, self.scaler, angle=angle,
                                                                         primitive=primitive, plot=plot)

            if terminal:
                critic_state = np.zeros(clutter.RealRealState.dim())
            else:
                state_ = clutter.preprocess_real_state(state, self.max_init_distance, angle=angle,
                                                       primitive=primitive)
                critic_state = clutter.RealRealState(state_, angle=0, sort=True, normalize=True, spherical=True,
                                                 range_norm=[-1, 1],
                                                 translate_wrt_target=False)
                if plot:
                    critic_state.plot()
                    plt.show()

                critic_state = critic_state.array()

        elif self.asymmetric == 'visual':
            actor_state = clutter.get_asymmetric_actor_feature_from_dict(state, self.ae, self.scaler, angle=angle,
                                                                         primitive=primitive, plot=plot)

            critic_state = clutter.get_asymmetric_actor_feature_from_dict(state, self.ae, self.scaler, angle=angle,
                                                                          primitive=primitive, plot=plot)

        elif self.asymmetric == 'real':
            if terminal:
                actor_state = np.zeros(clutter.RealRealState.dim())
                critic_state = np.zeros(clutter.RealRealState.dim())
            else:
                feature = clutter.preprocess_real_state(state, self.max_init_distance, angle=angle,
                                                        primitive=primitive)
                actor_state = clutter.RealRealState(feature, angle=0, sort=True, normalize=True, spherical=True,
                                                range_norm=[-1, 1], translate_wrt_target=False)
                state_ = clutter.preprocess_real_state(state, self.max_init_distance, angle=angle,
                                                       primitive=primitive)
                critic_state = clutter.RealRealState(state_, angle=0, sort=True, normalize=True, spherical=True,
                                                 range_norm=[-1, 1],
                                                 translate_wrt_target=False)

                if plot:
                    actor_state.plot()
                    plt.show()
                    critic_state.plot()
                    plt.show()

                actor_state = actor_state.array()
                critic_state = critic_state.array()

        else:
            raise ValueError

        return actor_state, critic_state

def train_eval(params):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='pushtarget_real_real', file_level=logging.INFO)
    agent = SplitDDPG(None, None, params=params['agent']['params'])
    trainer = TrainEvalWorld(agent=agent, env=params['env'],
                             params={'episodes': 10000,
                                     'eval_episodes': 20,
                                     'eval_every': 100,
                                     'eval_render': False,
                                     'save_every': 100})
    trainer.seed(0)
    trainer.run()
    print('Logging dir:', params['world']['logging_dir'])

def eval_with_render(dir):
    rb_logging.init(directory='/tmp/robamine_logs', friendly_name='', file_level=logging.INFO)
    with open(os.path.join(dir, 'config.yml'), 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config['env']['params']['render'] = True
    config['env']['params']['push']['predict_collision'] = False
    config['env']['params']['max_timesteps'] = 10
    config['env']['params']['nr_of_obstacles'] = [8, 13]
    config['env']['params']['safe'] = False
    config['env']['params']['log_dir'] = '/tmp'
    config['world']['episodes'] = 10
    world = EvalWorld.load(dir, overwrite_config=config)
    # world.seed_list = np.arange(33, 40, 1).tolist()
    world.seed(100)
    world.run()


if __name__ == '__main__':
    pid = os.getpid()
    print('Process ID:', pid)
    hostname = socket.gethostname()

    yml_name = 'params.yml'
    if hostname == 'dream':
        logging_dir = '/home/espa/robamine_logs/'
    elif hostname == 'triss':
        logging_dir = '/home/iason/robamine_logs/2020.01.16.split_ddpg/'
    elif hostname == 'iti-479':
        logging_dir = '/home/mkiatos/robamine/logs/'
    else:
        raise ValueError()

    with open(yml_name, 'r') as stream:
        params = yaml.safe_load(stream)

    # logging_dir = '/tmp'
    params['world']['logging_dir'] = logging_dir
    params['env']['params']['vae_path'] = os.path.join(logging_dir, 'VAE')
    params['agent']['params']['actor']['autoencoder']['model'] = os.path.join(logging_dir, 'VAE/model.pkl')
    params['agent']['params']['actor']['autoencoder']['scaler'] = os.path.join(logging_dir, 'VAE/normalizer.pkl')
    params['agent']['params']['n_preloaded_buffer'] = [500, 500]
    train_eval(params)

