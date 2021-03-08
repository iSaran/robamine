from robamine.algo.core import EvalWorld, TrainEvalWorld
from robamine import rb_logging
import logging
import yaml
import socket
import numpy as np
import os
import gym
import copy
import robamine.algo.core as core
import robamine.algo.util as algo_util
import torch
import torch.nn as nn
import robamine.algo.conv_vae as ae
import torch.optim as optim
import robamine.utils.memory as rb_mem
import robamine.envs.clutter_utils as clutter
import pickle
import robamine.algo.splitddpg as ddpg
import math

logger = logging.getLogger('robamine')

from robamine.experiments.ral.supervised_push_obstacle import Actor, PushObstacleRealPolicyDeterministic, ObsDictPolicy

class ReplayBuffer(rb_mem.ReplayBuffer):
    def sample_batch(self, given_batch_size):
        batch = []

        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch = self.random.sample(self.buffer, batch_size)

        state_batch = np.array([_.state for _ in batch])
        action_batch = np.array([_.action for _ in batch])
        reward_batch = np.array([_.reward for _ in batch])
        next_state_batch = []
        for i in range(len(batch[0].next_state)):
            next_state_batch.append(np.array([_.next_state[i] for _ in batch]))
        terminal_batch = np.array([_.terminal for _ in batch])

        return algo_util.Transition(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)

# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_units):
#         super(QNetwork, self).__init__()
#
#         self.hidden_layers = nn.ModuleList()
#         self.hidden_layers.append(nn.Linear(state_dim, hidden_units[0]))
#         i = 0
#         for i in range(1, len(hidden_units)):
#             self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
#
#         self.out = nn.Linear(hidden_units[i], action_dim)
#
#     def forward(self, x):
#         for layer in self.hidden_layers:
#             x = layer(x)
#             x = nn.functional.relu(x)
#         action_prob = self.out(x)
#         return action_prob
#
# class ObsDictPushTarget(ObsDictPolicy):
#     def __init__(self, nn, device='cpu'):
#         self.nn = nn
#         self.device = device
#
#     def predict(self, obs_dict):
#         state_ = obs_dict['push_target_feature'].copy()
#         state_ = torch.FloatTensor(state_).to(self.device)
#         action = self.nn(state_).detach().cpu().detach().numpy()
#         return np.insert(action, 0, 0)
#
#
# class ObsDictPushObstacle(ObsDictPolicy):
#     def __init__(self, actor):
#         self.actor = actor
#
#     def predict(self, obs_dict):
#         state_ = obs_dict['push_obstacle_feature'].copy()
#         return self.actor.predict(state_)
#
#
# class ComboFeature:
#     def __init__(self, ae, scaler):
#         self.ae = ae
#         self.scaler = scaler
#
#     def __call__(self, state, angle=0):
#         state_ = clutter.get_asymmetric_actor_feature_from_dict(state, self.ae, self.scaler, angle, primitive=0)
#         state_2 = clutter.get_asymmetric_actor_feature_from_dict(state, self.ae, None, angle, primitive=1)
#         return np.append(state_, state_2)
#
#
#
# class DQNCombo(core.RLAgent):
#     def __init__(self, params, push_target_actor, push_obstacle_actor, seed):
#         torch.manual_seed(seed)
#         self.state_dim = clutter.get_observation_dim(-1)
#         action_dim = 3
#         self.n_primitives = action_dim
#         super().__init__(self.state_dim, action_dim, 'DQN', params)
#         state_dim = 2 * (conv_ae.LATENT_DIM + 4)  # TODO: hardcoded the extra dim for surface edges
#
#         self.device = self.params['device']
#
#         self.network, self.target_network = QNetwork(state_dim, action_dim, self.params['hidden_units']).to(self.device), \
#                                             QNetwork(state_dim, action_dim, self.params['hidden_units']).to(self.device)
#         self.optimizer = optim.Adam(self.network.parameters(), lr=self.params['learning_rate'])
#         self.loss = nn.MSELoss()
#         self.replay_buffer = ReplayBuffer(self.params['replay_buffer_size'])
#         self.learn_step_counter = 0
#
#         self.rng = np.random.RandomState()
#
#         self.info['qnet_loss'] = 0
#         self.epsilon = self.params['epsilon_start']
#         self.info['epsilon'] = self.epsilon
#
#         self.policy = []
#         with open(push_target_actor, 'rb') as file:
#             pretrained_splitddpg = pickle.load(file)
#             actor = ddpg.Actor(int(state_dim / 2), pretrained_splitddpg['action_dim'][0], [400, 300])
#             actor.load_state_dict(pretrained_splitddpg['actor'][0])
#             self.policy.append(ObsDictPushTarget(actor, device=self.device))
#         self.policy.append(push_obstacle_actor)
#
#
#     def predict(self, state):
#         combo_feature = np.append(state['push_target_feature'], state['push_obstacle_feature'])
#         state_ = combo_feature
#         s = torch.FloatTensor(state_).to(self.device)
#         values = self.network(s).cpu().detach().numpy()
#         valid_nets, _ = clutter.get_valid_primitives(state, n_primitives=self.n_primitives)
#         values[valid_nets == False] = -1e6
#         primitive = np.argmax(values)
#         if primitive == 2:
#             return np.array([2])
#         action = self.policy[primitive].predict(state)
#         return action
#
#     def explore(self, state):
#         self.epsilon = self.params['epsilon_end'] + \
#                        (self.params['epsilon_start'] - self.params['epsilon_end']) * \
#                        math.exp(-1 * self.learn_step_counter / self.params['epsilon_decay'])
#         if self.rng.uniform(0, 1) >= self.epsilon:
#             return self.predict(state)
#
#         _, valid_nets = clutter.get_valid_primitives(state, n_primitives=self.n_primitives)
#         primitive = int(self.rng.choice(valid_nets, 1))
#         if primitive == 2:
#             return np.array([2])
#
#         action = self.policy[primitive].predict(state)
#         return action
#
#     def learn(self, transition):
#         self.info['qnet_loss'] = 0
#
#         transitions = self._transitions(transition)
#         for t in transitions:
#             self.replay_buffer.store(t)
#
#         # If we have not enough samples just keep storing transitions to the
#         # buffer and thus exit.
#         if self.replay_buffer.size() < self.params['batch_size']:
#             return
#
#         # Update target network if necessary
#         # if self.learn_step_counter > self.params['target_net_updates']:
#         #     self.target_network.load_state_dict(self.network.state_dict())
#         #     self.learn_step_counter = 0
#
#         new_target_params = {}
#         for key in self.target_network.state_dict():
#             new_target_params[key] = self.params['tau'] * self.target_network.state_dict()[key] + (
#                     1 - self.params['tau']) * self.network.state_dict()[key]
#         self.target_network.load_state_dict(new_target_params)
#
#         # Sample from replay buffer
#         batch = self.replay_buffer.sample_batch(self.params['batch_size'])
#         batch.terminal = np.array(batch.terminal.reshape((batch.terminal.shape[0], 1)))
#         batch.reward = np.array(batch.reward.reshape((batch.reward.shape[0], 1)))
#         batch.action = np.array(batch.action.reshape((batch.action.shape[0], 1)))
#
#         state = torch.FloatTensor(batch.state).to(self.device)
#         action = torch.LongTensor(batch.action.astype(int)).to(self.device)
#         next_state = torch.FloatTensor(batch.next_state).to(self.device)
#         terminal = torch.FloatTensor(batch.terminal).to(self.device)
#         reward = torch.FloatTensor(batch.reward).to(self.device)
#
#         if self.params['double_dqn']:
#             best_action = self.network(next_state).max(1)[1]  # action selection
#             q_next = self.target_network(next_state).gather(1, best_action.view(self.params['batch_size'],
#                                                                                 1))  # action evaluation
#         else:
#             q_next = self.target_network(next_state).max(1)[0].view(self.params['batch_size'], 1)
#
#         q_target = reward + (1 - terminal) * self.params['discount'] * q_next
#         q = self.network(state).gather(1, action)
#         loss = self.loss(q, q_target)
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         self.learn_step_counter += 1
#         self.info['qnet_loss'] = loss.detach().cpu().numpy().copy()
#         self.info['epsilon'] = self.epsilon
#
#     @classmethod
#     def load(cls, file_path, push_target_actor, push_obstacle_actor, seed):
#         model = pickle.load(open(file_path, 'rb'))
#         params = model['params']
#         self = cls(params, push_target_actor, push_obstacle_actor, seed)
#         self.load_trainable(model)
#         self.learn_step_counter = model['learn_step_counter']
#         logger.info('Agent loaded from %s', file_path)
#         return self
#
#     def load_trainable(self, input):
#         if isinstance(input, dict):
#             trainable = input
#             logger.warn('Trainable parameters loaded from dictionary.')
#         elif isinstance(input, str):
#             trainable = pickle.load(open(input, 'rb'))
#             logger.warn('Trainable parameters loaded from: ' + input)
#         else:
#             raise ValueError('Dict or string is valid')
#
#         self.network.load_state_dict(trainable['network'])
#         self.target_network.load_state_dict(trainable['target_network'])
#
#     def save(self, file_path):
#         model = {}
#         model['params'] = self.params
#         model['network'] = self.network.state_dict()
#         model['target_network'] = self.target_network.state_dict()
#         model['learn_step_counter'] = self.learn_step_counter
#         model['state_dim'] = self.state_dim
#         model['action_dim'] = self.action_dim
#         pickle.dump(model, open(file_path, 'wb'))
#
#     def q_value(self, state, action):
#         state_ = np.append(state['push_target_feature'], state['push_obstacle_feature'])
#         s = torch.FloatTensor(state_).to(self.device)
#         return self.network(s).cpu().detach().numpy()[int(action[0])]
#
#     def seed(self, seed):
#         self.replay_buffer.seed(seed)
#         self.rng.seed(seed)
#
#     def _transitions(self, transition):
#         transitions = []
#
#         # Create rotated states if needed
#         transition.state['init_distance_from_target'] = transition.next_state['init_distance_from_target']
#         state = np.append(transition.state['push_target_feature'], transition.state['push_obstacle_feature'])
#         next_state = np.append(transition.next_state['push_target_feature'], transition.next_state['push_obstacle_feature'])
#
#         # statee = {}
#         # statee['real_state'] = copy.deepcopy(real_state)
#         # statee['point_cloud'] = copy.deepcopy(point_cloud)
#         # next_statee = {}
#         # next_statee['real_state'] = copy.deepcopy(real_state_next)
#         # next_statee['point_cloud'] = copy.deepcopy(point_cloud_next)
#         tran = algo_util.Transition(state=copy.deepcopy(state),
#                                     action=transition.action[0],
#                                     reward=transition.reward,
#                                     next_state=copy.deepcopy(next_state),
#                                     terminal=transition.terminal)
#         transitions.append(tran)
#         return transitions
#
#



class Episode(core.Episode):
    def run(self, render=False, init_state=None, seed=None):
        print('Run episode with seed:', seed)
        state = self.env.reset(seed=seed)
        # self.env.load_state_dict(init_state)

        while True:
            if (render):
                self.env.render()
            action = self._action_policy(state)

            next_state, reward, done, info = self.env.step(action)
            if info['empty']:
                self.agent.last_was_empty = True
            print('Run episode with seed:', seed, 'action:', action, 'reward:', reward, 'done:', done,
                  'termination_reason:', info['termination_reason'])
            transition = algo_util.Transition(state, action, reward, next_state, done)
            self._learn(transition)
            self._update_stats_step(transition, info)

            state = next_state.copy()
            if done:
                self.termination_reason = info['termination_reason']
                break

        self._update_states_episode(info)


class TestingEpisode(Episode):
    def _action_policy(self, state):
        return self.agent.predict(state)

    def _learn(self, transition):
        pass

class EvalWorld2(core.EvalWorld):
    def run_episode(self, i):
        episode = TestingEpisode(self.agent, self.env)

        # LIke super run episode
        if self.env_init_states:
            init_state = self.env_init_states[i]
        else:
            init_state = None

        if self.seed_list is not None:
            seed = self.seed_list[i + self.seed_offset]
        else:
            seed = self.rng.randint(0, 999999999)

        episode.run(render=self.render, init_state=init_state, seed=seed)

        # Update tensorboard stats
        self.stats.update(i, episode.stats)
        self.episode_stats.append(episode.stats)
        self.episode_list_data.append(episode.data)

        # Save agent model
        self.save()
        if self.save_every and (i + 1) % self.save_every == 0:
            self.save(suffix='_' + str(i+1))

        # Save the config in YAML file
        self.experience_time += episode.stats['experience_time']
        self.update_results(n_iterations = i + 1, n_timesteps = episode.stats['n_timesteps'])

        for i in range (0, episode.stats['n_timesteps']):
            self.expected_values_file.write(str(episode.stats['q_value'][i]) + ',' + str(episode.stats['monte_carlo_return'][i]) + '\n')
            self.expected_values_file.flush()

        for i in range(len(episode.stats['actions_performed']) - 1):
            self.actions_file.write(str(episode.stats['actions_performed'][i]) + ',')
        self.actions_file.write(str(episode.stats['actions_performed'][-1]) + '\n')
        self.actions_file.flush()

        print('---')
        self.episode_list_data.calc()
        print(self.episode_list_data.__str__())


class ObsDictPushTarget(ObsDictPolicy):
    def __init__(self, nn, device='cpu'):
        self.nn = nn
        self.device = device

    def predict(self, obs_dict):
        state_ = obs_dict['push_target_feature'].copy()
        state_ = torch.FloatTensor(state_).to(self.device)
        action = self.nn(state_).detach().cpu().detach().numpy()
        return np.insert(action, 0, 0)

class ObsDictPushObstacle(ObsDictPolicy):
    def __init__(self, actor):
        self.actor = actor

    def predict(self, obs_dict):
        state_ = obs_dict['push_obstacle_feature'].copy()
        angle = float(self.actor.predict(state_))
        return np.array([1, angle])

class ObsDictSlideTarget(ObsDictPolicy):
    def predict(self, obs_dict):
        return np.array([2])


def split_replay_buffer(buffer, nr_buffers, nr_substates):
    """ Splits a buffer with mixed transitions (from different primitives) to
    one buffer per primitive.
    """
    result = []
    for _ in range(nr_buffers):
        result.append(ReplayBuffer(1e6))
    for i in range(buffer.size()):
        result[int(np.floor(buffer(i).action / nr_substates))].store(buffer(i))
    return result

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units):
        super(QNetwork, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(state_dim, hidden_units[0]))
        i = 0
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            self.hidden_layers[i].weight.data.uniform_(-0.003, 0.003)
            self.hidden_layers[i].bias.data.uniform_(-0.003, 0.003)

        self.out = nn.Linear(hidden_units[i], action_dim)

        self.out.weight.data.uniform_(-0.003, 0.003)
        self.out.bias.data.uniform_(-0.003, 0.003)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = nn.functional.relu(x)
        action_prob = self.out(x)
        return action_prob

class SplitDQN(core.RLAgent):
    def __init__(self, params, push_target_actor, push_obstacle_actor, seed):
        super().__init__(None, None, 'SplitDQN', params)
        torch.manual_seed(seed)

        self.nr_network = 2
        state_dim = [ae.LATENT_DIM + 4, ae.LATENT_DIM]

        self.device = self.params['device']

        # Create a list of networks and their targets
        self.network, self.target_network = nn.ModuleList(), nn.ModuleList()
        hidden = self.params['hidden_units']
        for i in range(len(hidden)):
            self.network.append(QNetwork(state_dim[i], 1, hidden[i]).to(self.device))
            self.target_network.append(QNetwork(state_dim[i], 1, hidden[i]).to(self.device))

        self.optimizer, self.replay_buffer, self.loss = [], [], []
        for i in range(self.nr_network):
            self.optimizer.append(optim.Adam(self.network[i].parameters(), lr=self.params['learning_rate'][i]))
            self.replay_buffer.append(ReplayBuffer(self.params['replay_buffer_size']))
            if self.params['loss'][i] == 'mse':
                self.loss.append(nn.MSELoss())
            elif self.params['loss'][i] == 'huber':
                self.loss.append(nn.SmoothL1Loss())
            else:
                raise ValueError('SplitDQN: Loss should be mse or huber')
            self.info['qnet_' +  str(i) + '_loss'] = 0

        self.learn_step_counter = 0
        self.rng = np.random.RandomState()
        self.rng.seed(seed)
        self.epsilon = self.params['epsilon_start']

        self.policy = [push_target_actor, push_obstacle_actor]
        self.last_was_empty = False

    def predict(self, state):

        if self.last_was_empty:
            action = self.policy[0].predict(state)
            self.last_was_empty = False
            return action

        valid_nets, _ = clutter.get_valid_primitives(state, n_primitives=self.nr_network)
        values = - 1e6 * np.ones(self.nr_network)

        state_feature = [state['push_target_feature'].copy(), state['push_obstacle_feature'].copy()]
        print('push obstacle feature:', state['push_obstacle_feature'])
        print('min max', np.min(state['push_obstacle_feature']), np.max(state['push_obstacle_feature']))

        for i in range(self.nr_network):
            s = torch.FloatTensor(state_feature[i]).to(self.device)
            if valid_nets[i]:
                values[i] = float(self.network[i](s).cpu().detach().numpy())
        primitive = np.argmax(values)
        x = input('action:')
        primitive = int(x)
        action = self.policy[primitive].predict(state)
        return action

    def explore(self, state):
        self.epsilon = self.params['epsilon_end'] + \
                       (self.params['epsilon_start'] - self.params['epsilon_end']) * \
                       math.exp(-1 * self.learn_step_counter / self.params['epsilon_decay'])
        if self.rng.uniform(0, 1) >= self.epsilon:
            result = self.predict(state)
            return result

        _, valid_nets = clutter.get_valid_primitives(state, n_primitives=self.nr_network)
        primitive = int(self.rng.choice(valid_nets, 1))
        action = self.policy[primitive].predict(state)
        return action

    def learn(self, transition):
        i = int(transition.action[0])
        self.replay_buffer[i].store(self._transitions(transition))

        for _ in range(self.params['update_iter'][i]):
            self.update_net(i)

    def update_net(self, i):
        self.info['qnet_' +  str(i) + '_loss'] = 0

        # If we have not enough samples just keep storing transitions to the
        # buffer and thus exit.
        if self.replay_buffer[i].size() < self.params['n_preloaded_buffer'][i]:
            return

        # Update target net's params
        new_target_params = {}
        for key in self.target_network[i].state_dict():
            new_target_params[key] = self.params['tau'] * self.target_network[i].state_dict()[key] + (1 - self.params['tau']) * self.network[i].state_dict()[key]
        self.target_network[i].load_state_dict(new_target_params)

        # Sample from replay buffer
        batch = self.replay_buffer[i].sample_batch(self.params['batch_size'][i])
        batch.terminal = np.array(batch.terminal.reshape((batch.terminal.shape[0], 1)))
        batch.reward = np.array(batch.reward.reshape((batch.reward.shape[0], 1)))

        # Calculate maxQ(s_next, a_next) with max over next actions
        q_next = torch.FloatTensor().to(self.device)
        for net in range(self.nr_network):
            next_state = torch.FloatTensor(batch.next_state[net]).to(self.device)
            q_next_i = self.target_network[net](next_state)
            q_next = torch.cat((q_next, q_next_i), dim=1)
        q_next = q_next.max(1)[0].view(self.params['batch_size'][i], 1)

        reward = torch.FloatTensor(batch.reward).to(self.device)
        terminal = torch.FloatTensor(batch.terminal).to(self.device)
        q_target = reward + (1 - terminal) * self.params['discount'] * q_next

        # Calculate current q
        s = torch.FloatTensor(batch.state).to(self.device)
        q = self.network[i](s)

        loss = self.loss[i](q, q_target)
        self.optimizer[i].zero_grad()
        loss.backward()
        self.optimizer[i].step()
        self.info['qnet_' +  str(i) + '_loss'] = loss.detach().cpu().numpy().copy()

        self.learn_step_counter += 1

    def q_value(self, state, action):
        names = ['push_target_feature', 'push_obstacle_feature']
        i = int(action[0])
        s = torch.FloatTensor(state[names[i]]).to(self.device)
        return float(self.network[i](s).cpu().detach().numpy())

    def seed(self, seed):
        for i in range(len(self.replay_buffer)):
            self.replay_buffer[i].seed(seed)
        self.rng.seed(seed)

    @classmethod
    def load(cls, file_path, push_target_actor, push_obstacle_actor, seed=0):
        model = pickle.load(open(file_path, 'rb'))
        params = model['params']
        self = cls(params, push_target_actor, push_obstacle_actor, seed)
        self.load_trainable(model)
        self.learn_step_counter = model['learn_step_counter']
        logger.info('Agent loaded from %s', file_path)
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
            self.network[i].load_state_dict(trainable['network'][i])
            self.target_network[i].load_state_dict(trainable['target_network'][i])

    def save(self, file_path):
        model = {}
        model['params'] = self.params
        model['network'], model['target_network'] = [], []
        for i in range(self.nr_network):
            model['network'].append(self.network[i].state_dict())
            model['target_network'].append(self.target_network[i].state_dict())
        model['learn_step_counter'] = self.learn_step_counter
        pickle.dump(model, open(file_path, 'wb'))

    def _transitions(self, transition):
        # Create rotated states if needed
        names = ['push_target_feature', 'push_obstacle_feature']
        i = int(transition.action[0])
        state = transition.state[names[i]]
        next_state = []
        for name in names:
            next_state.append(transition.next_state[name])
        tran = algo_util.Transition(state=copy.deepcopy(state),
                                    action=transition.action[0],
                                    reward=transition.reward,
                                    next_state=copy.deepcopy(next_state),
                                    terminal=transition.terminal)
        return tran

class ComboExp:
    def __init__(self, params, push_target_actor_path, push_obstacle_actor_path, seed=0, friendly_name=''):
        self.params = copy.deepcopy(params)
        self.seed = seed
        self.friendly_name = friendly_name

        self.params['env']['params']['render'] = False
        self.params['env']['params']['target']['max_bounding_box'][2] = 0.01
        self.params['env']['params']['hardcoded_primitive'] = -1

        with open(push_target_actor_path, 'rb') as file:
            pretrained_splitddpg = pickle.load(file)
            actor = ddpg.Actor(ae.LATENT_DIM + 4, pretrained_splitddpg['action_dim'][0], [400, 300])
            actor.load_state_dict(pretrained_splitddpg['actor'][0])
        self.push_target_actor = ObsDictPushTarget(actor)


        if push_obstacle_actor_path == 'real':
            self.push_obstacle_actor = PushObstacleRealPolicyDeterministic()
        else:
            self.push_obstacle_actor = ObsDictPushObstacle(Actor.load(push_obstacle_actor_path))

        dqn_params = {'hidden_units': [[200, 200], [200, 200]],
                      'device': 'cpu',
                      'replay_buffer_size': 1000000,
                      'loss': ['mse', 'mse'],
                      'learning_rate': [1e-3, 1e-3],
                      'update_iter': [1, 1],
                      'n_preloaded_buffer': [4, 4],
                      'tau': 0.001,
                      'batch_size': [4, 4],
                      'discount': 0.99,

                      'epsilon_start': 0.9,
                      'epsilon_end': 0.05,
                      'epsilon_decay': 10000,
                      'heightmap_rotations': 1,
                      'double_dqn': True,
                      }

        self.agent = SplitDQN(params=dqn_params, push_target_actor=self.push_target_actor,
                              push_obstacle_actor=self.push_obstacle_actor, seed=self.seed)

    def train_eval(self, episodes=10000, eval_episodes=20, eval_every=100, save_every=100):
        rb_logging.init(directory=self.params['world']['logging_dir'], friendly_name=self.friendly_name,
                        file_level=logging.INFO)


        trainer = TrainEvalWorld(agent=self.agent, env=self.params['env'],
                                 params={'episodes': episodes,
                                         'eval_episodes': eval_episodes,
                                         'eval_every': eval_every,
                                         'eval_render': False,
                                         'save_every': save_every})
        trainer.seed(self.seed)
        trainer.run()
        print('Logging dir:', trainer.log_dir)

    def check_transition(self):
        '''Run environment'''

        self.params['env']['params']['render'] = True
        env = gym.make(self.params['env']['name'], params=self.params['env']['params'])

        while True:
            seed = np.random.randint(100000000)
            seed = 305097549
            print('Seed:', seed)
            rng = np.random.RandomState()
            rng.seed(seed)
            obs = env.reset(seed=seed)

            action = self.agent.explore(obs)
            print('predicted action', action)
            obs, reward, done, info = env.step(action)
            print('reward:', reward, 'action:', action, 'done:', done, 'temrination condiction:', info['termination_reason'])
            if done:
                break


    def eval_with_render(self):
        rb_logging.init(directory='/tmp/robamine_logs', friendly_name='', file_level=logging.INFO)
        directory = os.path.join(self.params['world']['logging_dir'], self.friendly_name)
        with open(os.path.join(directory, 'config.yml'), 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        agent = SplitDQN.load(os.path.join(directory, 'model.pkl'), self.push_target_actor, self.push_obstacle_actor,
                              self.seed)
        config['env']['params']['render'] = True
        config['env']['params']['push']['predict_collision'] = False
        config['env']['params']['max_timesteps'] = 10
        config['env']['params']['log_dir'] = '/tmp'
        config['env']['params']['deterministic_policy'] = True
        config['env']['params']['nr_of_obstacles'] = [11, 13]
        config['env']['params']['target']['randomize_pos'] = False
        config['world']['episodes'] = 20
        world = EvalWorld(agent, env=config['env'], params=config['world'])
        world.seed_list = np.arange(0, 10, 1).tolist()
        world.run()

    def eval_in_scenes(self, n_scenes=1000):
        rb_logging.init(directory=params['world']['logging_dir'], friendly_name=self.friendly_name + '_eval_test', file_level=logging.INFO)
        directory = os.path.join(self.params['world']['logging_dir'], self.friendly_name)
        with open(os.path.join(directory, 'config.yml'), 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        config['env']['params']['render'] = False
        config['env']['params']['push']['predict_collision'] = True
        config['env']['params']['log_dir'] = params['world']['logging_dir']
        config['env']['params']['deterministic_policy'] = True
        config['env']['params']['nr_of_obstacles'] = [8, 13]
        config['env']['params']['obstacle']['pushable_threshold_coeff'] = 1
        config['env']['params']['target']['randomize_pos'] = False
        # config['env']['params']['target']['min_bounding_box'] = [.03, .02, .005]
        # config['env']['params']['target']['max_bounding_box'] = [.03, .02, .020]
        # config['env']['params']['obstacle']['min_bounding_box'] = [.03, .02, .005]
        # config['env']['params']['obstacle']['max_bounding_box'] = [.03, .02, .020]
        # config['env']['params']['hug_probability'] = 0.0
        config['world']['episodes'] = n_scenes
        agent = SplitDQN.load(os.path.join(directory, 'model.pkl'), self.push_target_actor, self.push_obstacle_actor,
                              self.seed)
        world = EvalWorld2(agent, env=config['env'], params=config['world'])
        world.seed_list = np.arange(0, n_scenes, 1).tolist()
        world.run()
        print('Logging dir:', params['world']['logging_dir'])

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

    exp = ComboExp(params=params,
                   push_target_actor_path=os.path.join(logging_dir, '../ral-results/env-very-hard/splitac-modular/push-target/train/model.pkl'),
                   push_obstacle_actor_path=os.path.join(logging_dir, 'push_obstacle_supervised/actor_deterministic_256size/model_60.pkl'),
                   # push_obstacle_actor_path='real',
                   friendly_name='combo_split_dqn_deterministic_256size',
                   seed=1)
    # exp.train_eval(episodes=10000, eval_episodes=20, eval_every=100, save_every=100)
    exp.eval_with_render()
    # exp.check_transition()
    # exp.eval_in_scenes(n_scenes=1000)
