from robamine.algo.core import TrainWorld, EvalWorld, TrainEvalWorld, SupervisedTrainWorld
# from robamine.algo.ddpg_torch import DDPG_TORCH
from robamine.algo.splitddpg import SplitDDPG, Critic
from robamine.algo.util import EpisodeListData
from robamine.algo.core import Agent, RLAgent
from robamine import rb_logging
import logging
import yaml
import sys
import socket
import numpy as np
import os
import gym
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from robamine.algo.util import get_agent_handle
import robamine.envs.clutter_utils as clutter
import matplotlib.pyplot as plt
from robamine.utils.math import min_max_scale
from robamine.utils.memory import get_batch_indices
from math import pi
from math import floor


logger = logging.getLogger('robamine')


# General functions for training and rendered eval
# ------------------------------------------------

def train(params):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='', file_level=logging.INFO)
    trainer = TrainWorld(agent=params['agent'], env=params['env'], params=params['world']['params'])
    trainer.run()
    print('Logging dir:', params['world']['logging_dir'])

def train_eval(params):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='', file_level=logging.INFO)
    trainer = TrainEvalWorld(agent=params['agent'], env=params['env'],
                             params={'episodes': 10000,
                                     'eval_episodes': 20,
                                     'eval_every': 100,
                                     'eval_render': False,
                                     'save_every': 100})
    trainer.seed(21321)
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
    config['env']['params']['max_timesteps'] = 5
    config['env']['params']['safe'] = False
    config['env']['params']['log_dir'] = '/tmp'
    config['world']['episodes'] = 10
    world = EvalWorld.load(dir, overwrite_config=config)
    # world.seed_list = np.arange(33, 40, 1).tolist()
    world.seed(100)
    world.run()

def eval_in_scenes(params, dir, n_scenes=1000):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='', file_level=logging.INFO)
    with open(os.path.join(dir, 'config.yml'), 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config['env']['params']['render'] = False
    config['env']['params']['safe'] = False
    config['env']['params']['log_dir'] = params['world']['logging_dir']
    config['env']['params']['deterministic_policy'] = True
    config['world']['episodes'] = n_scenes
    world = EvalWorld.load(dir, overwrite_config=config)
    world.seed_list = np.arange(0, n_scenes, 1).tolist()
    world.run()
    print('Logging dir:', params['world']['logging_dir'])

def analyze_multiple_eval_envs(dir_, results_dir):
    exps = [
        {'name': 'SplitAC-scr', 'path': '../ral-results/env-icra/splitac-scratch', 'action_discrete': False},
        {'name': 'SplitDQN', 'path': '../ral-results/env-icra/splitdqn-3', 'action_discrete': True},
        {'name': 'SplitDQN-13', 'path': '../ral-results/env-very-hard/splitdqn', 'action_discrete': True}]
    for i in range(len(exps)):
        exps[i]['path'] = os.path.join(dir_, exps[i]['path'])
    analyze_multiple_evals(exps, results_dir, env_name='Env-ICRA')

    exps = [
        {'name': 'Random-Cont', 'path': '../ral-results/env-hard/random-cont', 'action_discrete': False},
        {'name': 'SplitAC-scr', 'path': '../ral-results/env-hard/splitac-scratch', 'action_discrete': False},
        {'name': 'SplitDQN', 'path': '../ral-results/env-hard/splitdqn', 'action_discrete': True},
        {'name': 'SplitAC-pt', 'path': '../ral-results/env-hard/splitac-modular/push-target', 'action_discrete': False},
        {'name': 'Push-Target', 'path': '../ral-results/env-hard/splitac-modular/push-target', 'action_discrete': False},
        {'name': 'Push-Obstacle', 'path': '../ral-results/env-hard/splitac-modular/push-obstacle', 'action_discrete': False},
        {'name': 'SplitAC-combo', 'path': '../ral-results/env-hard/splitac-modular/combo', 'action_discrete': False}]
    for i in range(len(exps)):
        exps[i]['path'] = os.path.join(dir_, exps[i]['path'])
    analyze_multiple_evals(exps, results_dir, env_name='Env-Hard')

    exps = [{'name': 'Random-Cont', 'path': '../ral-results/env-very-hard/random-cont', 'action_discrete': False}]
    for i in range(len(exps)):
        exps[i]['path'] = os.path.join(dir_, exps[i]['path'])
    analyze_multiple_evals(exps, results_dir, env_name='Env-very-hard')

def analyze_multiple_evals(exps, results_dir, env_name='Metric'):
    names, dirs = [], []
    for i in range(len(exps)):
        names.append(exps[i]['name'])
        dirs.append(exps[i]['path'])

    from tabulate import tabulate
    import seaborn
    import pandas as pd
    # seaborn.set(style="whitegrid")
    seaborn.set(style="ticks", palette="pastel")


    headers = [env_name]
    column_0 = ['Valid Episodes',
               'Singulation in 5 steps for valid episodes %',
               'Singulation in 10 steps for valid episodes %',
               'Singulation in 15 steps for valid episodes %',
               'Singulation in 20 steps for valid episodes %',
               'Fallen %',
               'Max timesteps terminals %',
               'Collision terminals %',
               'Flips terminals %',
               'Empty terminals %',
               'Mean reward per step',
               'Mean actions for singulation',
               'Push target used %',
               'Push Obstacle used %',
               'Extra primitive used %',
               'Model trained for (timesteps)']

    percentage = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14]

    data = [None] * len(names)
    columns = [None] * len(names)
    table = []
    for i in range(len(names)):
        headers.append(names[i])
        data[i], columns[i] = analyze_eval_in_scenes(dirs[i], action_discrete=exps[i]['action_discrete'])

    for i in range(len(column_0)):
        row = []
        row.append(column_0[i])
        for j in range(len(names)):
            if i in percentage:
                row.append(columns[j][i] * 100)
            else:
                row.append(columns[j][i])

        table.append(row)
    pd.DataFrame(table, columns=headers).to_csv(os.path.join(results_dir, env_name + '.csv'), index=False)
    print('')
    print(tabulate(table, headers=headers))

    # Violin plots
    fig, axes = plt.subplots()
    df = pd.DataFrame({names[0]: data[0]})
    for i in range(1, len(names)):
        df = pd.concat([df, pd.DataFrame({names[i]: data[i]})], axis=1)

    seaborn.violinplot(data=df, bw=.4, cut=2, scale='area',
                       linewidth=1, inner='box', orient='h')
    plt.axvline(x=5, color='gray', linestyle='--')

    # seaborn.violinplot(data=data[1], palette="Set3", bw=.2, cut=2,
    #                    linewidth=3)
    # seaborn.violinplot(data=data[2], palette="Set3", bw=.2, cut=2,
    #                    linewidth=3)

    # axes.violinplot(data)
    # axes.violinplot(data, [0], points=100, widths=0.3,
    #                 showmeans=True, showextrema=True, showmedians=True)
    plt.savefig(os.path.join(results_dir, env_name + '.png'))



def analyze_eval_in_scenes(dir, action_discrete=False):
    from collections import OrderedDict
    training_timesteps = 0
    path = os.path.join(dir, 'train/episodes')
    if not os.path.exists(path):
        training_timesteps = np.nan
    else:
        training_data = EpisodeListData.load(path)
        for i in range(len(training_data)):
            training_timesteps += len(training_data[i])

    data = EpisodeListData.load(os.path.join(dir, 'eval/episodes'))
    singulations, fallens, collisions, timesteps = 0, 0, 0, 0
    steps_singulations = []
    episodes = len(data)
    rewards = []
    timestep_terminals = 0
    collision_terminals = 0
    deterministic_collision_terminals = 0
    flips = 0
    episodes_terminated = 0
    empties = 0
    push_target_used = 0
    push_obstacle_used = 0
    extra_primitive_used = 0

    under = [5, 10, 15, 20]
    singulation_under = OrderedDict()
    for k in under:
        singulation_under[k] = 0

    for i in range(episodes):
        timesteps += len(data[i])
        if data[i][-1].transition.terminal:
            episodes_terminated += 1
            if data[i][-1].transition.info['termination_reason'] == 'singulation':
                for k in under:
                    if len(data[i]) <= k:
                        singulation_under[k] += 1
                singulations += 1
                steps_singulations.append(len(data[i]))
            elif data[i][-1].transition.info['termination_reason'] == 'fallen':
                fallens += 1
            elif data[i][-1].transition.info['termination_reason'] == 'timesteps':
                timestep_terminals += 1
            elif data[i][-1].transition.info['termination_reason'] == 'collision':
                collision_terminals += 1
            elif data[i][-1].transition.info['termination_reason'] == 'deterministic_collision':
                deterministic_collision_terminals += 1
            elif data[i][-1].transition.info['termination_reason'] == 'flipped':
                episodes_terminated -= 1
                flips += 1
            elif data[i][-1].transition.info['termination_reason'] == 'empty':
                empties += 1
            else:
                raise Exception(data[i][-1].transition.info['termination_reason'])

            for j in range(len(data[i])):
                rewards.append(data[i][j].transition.reward)

        for timestep in range(len(data[i])):
            if action_discrete:
                if data[i][timestep].transition.action < 8:
                    push_target_used += 1
                elif data[i][timestep].transition.action < 16:
                    push_obstacle_used += 1
                else:
                    extra_primitive_used += 1
            else:
                if data[i][timestep].transition.action[0] == 0:
                    push_target_used += 1
                elif data[i][timestep].transition.action[0] == 1:
                    push_obstacle_used += 1

    # print('terminal singulations:', (singulations / episodes) * 100, '%')
    # print('terminal fallens:', (fallens / episodes) * 100, '%')
    # print('collisions:', (collisions / episodes) * 100, '%')
    # print('Total timesteps:', timesteps)
    # print('Mean steps for singulation:', np.mean(steps_singulations))
    # plt.hist(steps_singulations)
    # plt.show()
    # plt.hist(rewards)
    # plt.show()
    # data = sorted(steps_singulations)
    # print('25th perc:', data[int(0.25 * len(data))])
    # print('50th perc:', data[int(0.5 * len(data))])
    # print('75th perc:', data[int(0.75 * len(data))])


    for k in under:
        singulation_under[k] /= episodes_terminated

    results = [episodes_terminated,
               singulation_under[5],
               singulation_under[10],
               singulation_under[15],
               singulation_under[20],
               (fallens / episodes),
               (timestep_terminals / episodes),
               (collision_terminals / episodes),
               (flips / episodes),
               (empties / episodes),
               np.mean(rewards),
               np.mean(steps_singulations),
               push_target_used / timesteps,
               push_obstacle_used / timesteps,
               extra_primitive_used / timesteps,
               training_timesteps]

    return steps_singulations, results



def process_episodes(dir):
    data = EpisodeListData.load(os.path.join(dir, 'episodes'))
    data.calc()
    print('Nr of episodes:', len(data))
    print('Success rate', data.success_rate)
    for episode in data:
        print(episode.actions_performed)

def check_transition(params):
    '''Run environment'''

    # Load autoencoder
    # import robamine.algo.conv_vae as ae
    # with open('/home/iason/robamine_logs/2020.01.16.split_ddpg/VAE/model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    # latent_dim = model['encoder.fc.weight'].shape[0]
    # vae = ae.ConvVae(latent_dim)
    # vae.load_state_dict(model)

    params['env']['params']['render'] = True
    params['env']['params']['safe'] = False
    env = gym.make(params['env']['name'], params=params['env']['params'])

    # seed = 39574883  # seed for scene with obstacles visible
    # seed = 28794391
    # seed = 77269035
    # f = RealState(obs, normalize=True)
    # print(f.array()) print(f.principal_corners)

    # print

    # RealState(obs['heightmap_mask'][0]).plot()
    # # PushTargetFeature(obs).plot()
    #
    #
    #
    #
    # action = np.array([0, 1, 0, 1])
    while True:
        seed = np.random.randint(100000000)
        # seed = 39574883  # seed for scene with obstacles visible
        # seed = 28794391
        # seed = 77269035
        # seed = 17176674
        # seed = 16193743
        print('Seed:', seed)
        rng = np.random.RandomState()
        rng.seed(seed)
        obs = env.reset(seed=seed)
        # RealState(obs, spherical=True)
        # plot_point_cloud_of_scene(obs)
        # get_asymmetric_actor_feature_from_dict(obs, vae, None, angle=0, plot=True)

        while True:
            action = rng.uniform(-1, 1, 4)
            action[0] = 0
            # action_ = np.zeros(action.shape)
            # action_[1] = min_max_scale(action[1], range=[-1, 1], target_range=[-pi, pi])
            # action_[2] = min_max_scale(action[2], range=[-1, 1], target_range=params['env']['params']['push']['distance'])
            # action_[3] = min_max_scale(action[3], range=[-1, 1], target_range=params['env']['params']['push']['target_init_distance'])
            # RealState(obs).plot(action=action_)
            # action = [0, -1, 1, -1]
            obs, reward, done, info = env.step(action)
            print('reward:', reward, 'action:', action, 'done:', done, 'temrination condiction:', info['termination_reason'])
            # RealState(obs).plot()
            # array = RealState(obs).array()
            # if (array > 1).any() or (array < -1).any():
            #     print('out of limits indeces > 1:', array[array < -1])
            #     print('out of limits indeces < -1:', array[array < -1])
            # plot_point_cloud_of_scene(obs)
            if done:
                break

def test():
    from robamine.utils.math import cartesian2spherical
    points = np.array([[1, 1, 1], [2, 2, 2]])
    points = np.round(np.random.rand(5, 3), 1) * 10
    points[0] = np.zeros(3)
    points[1] = np.array([1, 2, 10])
    print('random', points)
    points = cartesian2shperical(points)
    print('final in spherical', points)
    points = cartesian2shperical(np.zeros(3))
    print(points)
    print(points.shape)

def visualize_critic_predictions(exp_dir, model_name='model.pkl'):
    # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    params['env']['params']['render'] = True
    params['env']['params']['safe'] = False
    params['env']['params']['push']['predict_collision'] = False
    env = gym.make(params['env']['name'], params=params['env']['params'])
    seed = np.random.randint(100000000)
    print('Seed:', seed)
    obs = env.reset(seed=seed)
    import math

    splitddpg = SplitDDPG.load(os.path.join(exp_dir, model_name))

    X = np.linspace(-1, 1, 10)
    Y = np.linspace(-1, 1, 10)
    U = np.linspace(-1, 1, 8)
    s = len(X)
    X, Y = np.meshgrid(X, Y)
    while(True):
        fig, axs = plt.subplots(len(U),)
        # print(X)
        # print(Y)
        Z = np.zeros((s, s))
        for k in range(len(U)):
            for i in range(s):
                for j in range(s):
                    Z[i, j] = splitddpg.q_value(obs, np.array([0, X[i, j], U[k], Y[i, j]]))

            co = axs[k].contourf(X, Y, Z)
            fig.colorbar(co, ax=axs[k])

        action = splitddpg.predict(obs)
        axs.scatter(action[1], action[2], color=[1, 0, 0])

            # axs[k].colorbar()

            # Z[i, j] = X[i, j] ** 2 + Y[i, j] ** 2

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(X, Y, Z)

        plt.show()

        obs, _, _, _ = env.step(action=action)

def train_combo_q_learning(params):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='', file_level=logging.INFO)
    from robamine.algo.splitddpg import DQNCombo

    agent = DQNCombo({'hidden_units': [100, 100],
                      'learning_rate': 1e-3,
                      'replay_buffer_size': 1000000,
                      'epsilon_start': 0.9,
                      'epsilon_end': 0.05,
                      'epsilon_decay': 10000,
                      'device': 'cpu',
                      'autoencoder_model': os.path.join(params['world']['logging_dir'], 'VAE/model.pkl'),
                      'autoencoder_scaler': os.path.join(params['world']['logging_dir'], 'VAE/normalizer.pkl'),
                      'pretrained': [os.path.join(params['world']['logging_dir'], 'splitcombo/push_target.pkl'), \
                                     os.path.join(params['world']['logging_dir'], 'splitcombo/push_obstacle.pkl')],
                      'heightmap_rotations': 8,
                      'batch_size': 32,
                      'tau': 0.001,
                      'double_dqn': True,
                      'discount': 0.99
                      })


    params['env']['params']['render'] = False
    params['env']['params']['hardcoded_primitive'] = -1
    env = gym.make(params['env']['name'], params=params['env']['params'])

    trainer = TrainWorld(agent=agent, env=env, params=params['world']['params'])
    trainer.run()


# Supervised learning of a critic
# -------------------------------
import math

class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(CriticNet, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_units[0]))
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            stdv = 1. / math.sqrt(self.hidden_layers[i].weight.size(1))
            self.hidden_layers[i].weight.data.uniform_(-stdv, stdv)
            self.hidden_layers[i].bias.data.uniform_(-stdv, stdv)

        self.out = nn.Linear(hidden_units[-1], 1)
        stdv = 1. / math.sqrt(self.out.weight.size(1))
        self.out.weight.data.uniform_(-stdv, stdv)
        self.out.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # Clone an reshape in case of single input in order to have a "batch" shape
        # x = torch.cat([x_, u_], x_.dim() - 1)
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        out = self.out(x)
        return out

class Critic(Agent):
    '''
    A class for train PyTorch networks. Inherit and create a self.network (which
    inherits from torch.nn.Module) before calling super().__init__()
    '''
    def __init__(self):
        super().__init__(name='Critic', params={})
        self.device = 'cpu'
        self.learning_rate = 0.001
        self.batch_size = 64
        # self.hidden_units = [800, 500, 300]
        self.hidden_units = [400, 300]

        self.network = CriticNet(125, self.hidden_units)

        # Create the networks, optimizers and loss
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=self.learning_rate)

        self.loss = nn.MSELoss()

        self.iterations = 0
        self.info['train'] = {'loss': 0.0}
        self.info['test'] = {'loss': 0.0}
        self.train_dataset = None
        self.test_dataset = None

    def load_dataset(self, dataset, split=0.8):
        self.dataset = dataset

        scenes = floor(split * len(dataset[2]))
        print('scenes', scenes)
        print(dataset[2][scenes])
        self.train_dataset = [dataset[0][:dataset[2][scenes]], dataset[1][:dataset[2][scenes]]]
        self.test_dataset = [dataset[0][dataset[2][scenes]:], dataset[1][dataset[2][scenes]:]]

    def predict(self, state):
        # ndim == 1 is assumed to mean 1 sample (not multiple samples of 1 feature)
        if state.ndim == 1:
            state_ = state.reshape(1, -1)
        else:
            state_ = state

        inputs = state_.copy()
        s = torch.FloatTensor(inputs).to(self.device)
        prediction = self.network(s).cpu().detach().numpy()

        return prediction

    def learn(self):
        '''Run one epoch'''
        self.iterations += 1

        # Calculate loss in train dataset
        train_x, train_y = self.train_dataset[0], self.train_dataset[1]
        real_x = torch.FloatTensor(train_x).to(self.device)
        prediction = self.network(real_x)
        real_y = torch.FloatTensor(train_y).to(self.device)
        loss = self.loss(prediction, real_y)
        self.info['train']['loss'] = loss.detach().cpu().numpy().copy()

        # Calculate loss in test dataset
        test_x, test_y = self.test_dataset[0], self.test_dataset[1]
        real_x = torch.FloatTensor(test_x).to(self.device)
        prediction = self.network(real_x)
        real_y = torch.FloatTensor(test_y).to(self.device)
        loss = self.loss(prediction, real_y)
        self.info['test']['loss'] = loss.detach().cpu().numpy().copy()

        # Minimbatch update of network
        minibatches = get_batch_indices(dataset_size=self.train_dataset[0].shape[0], batch_size=self.batch_size)
        for minibatch in minibatches:
            batch_x = self.train_dataset[0][minibatch]
            batch_y = self.train_dataset[1][minibatch]

            real_x = torch.FloatTensor(batch_x).to(self.device)
            prediction = self.network(real_x)
            real_y = torch.FloatTensor(batch_y).to(self.device)
            loss = self.loss(prediction, real_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['trainable'] = self.trainable_dict()
        state_dict['iterations'] = self.iterations
        return state_dict

    def trainable_dict(self):
        return self.network.state_dict()

    def load_trainable_dict(self, trainable):
        self.network.load_state_dict(trainable)

    def load_trainable(self, file_path):
        '''Assume that file path is a pickle with with self.state_dict() '''
        state_dict = pickle.load(open(input, 'rb'))
        self.load_trainable_dict(state_dict['trainable'])

    @classmethod
    def load_state_dict(cls, state_dict):
        self = cls()
        self.load_trainable_dict(state_dict['trainable'])
        self.iterations = state_dict['iterations']
        return self

    def seed(self, seed=None):
        super().seed(seed)
        self.train_dataset.seed(seed)
        self.test_dataset.seed(seed)

def collect_scenes_real_state(params, dir_to_save, n_scenes=1000):
    print('Collecting the real state of some scenes...')
    from robamine.envs.clutter_cont import ClutterContWrapper
    from robamine.envs.clutter_utils import InvalidEnvError

    def safe_seed_run(env):
        seed = np.random.randint(1000000000)
        try:
            obs = env.reset(seed=seed)
        except InvalidEnvError as e:
            print("WARN: {0}. Invalid environment during reset. A new environment will be spawn.".format(e))
            return safe_seed_run(env)
        return obs, seed

    params['env']['params']['render'] = False
    params['env']['params']['safe'] = False
    env = ClutterContWrapper(params=params['env']['params'])

    real_states = []
    keywords = ['object_poses', 'object_above_table', 'object_bounding_box', 'max_n_objects', 'surface_size',
                'finger_height', 'n_objects']

    for i in range(n_scenes):
        print('Collecting scenes. Scene: ', i, 'out of', n_scenes)
        scene = {}

        obs, seed = safe_seed_run(env)
        scene['seed'] = seed

        for key in keywords:
            scene[key] = obs[key]
        real_states.append(scene)
        plt.imsave(os.path.join(dir_to_save, 'screenshots/' + str(i) + '_screenshot.png'), env.env.bgr)

    with open(os.path.join(dir_to_save, 'scenes.pkl'), 'wb') as file:
        pickle.dump([real_states, params], file)

def create_dataset_from_scenes(dir, n_x=16, n_y=16):
    from robamine.envs.clutter_utils import predict_collision

    def reward(obs_dict, p, distance):
        if predict_collision(obs_dict, p[0], p[1]):
            return -1

        reward = 1 - min_max_scale(distance, range=[-1, 1], target_range=[0, 1])
        return reward

    with open(os.path.join(dir, 'scenes.pkl'), 'rb') as file:
        data, params = pickle.load(file)


    x = np.linspace(-1, 1, n_x)
    y = np.linspace(-1, 1, n_y)
    x, y = np.meshgrid(x, y)
    x_y = np.column_stack((x.ravel(), y.ravel()))
    distance = np.linalg.norm(x_y[0, :] - x_y[1, :])
    x_y = x_y[np.linalg.norm(x_y, axis=1) <= 1]

    n_scenes = len(data)
    n_datapoints = n_scenes * x_y.shape[0]
    n_features = 125
    dataset_x = np.zeros((n_datapoints, n_features))
    dataset_y = np.zeros((n_datapoints, 1))

    # for scene in data:
    # r = np.linspace(-1, 1, n_actions_r)
    # theta = np.linspace(-1, 1, n_actions_theta)
    # r, theta = np.meshgrid(r, theta)
    sample = 0
    scene_start_id = np.zeros(len(data), dtype=np.int32)
    for scene in range(len(data)):
        print('Creating dataset scenes. Scene: ', scene, 'out of', n_scenes)
        x_y_random = np.random.normal(x_y, distance / 4)
        plt.plot(x_y_random[:, 0], x_y_random[:, 1], marker='.', color='k', linestyle='none')
        plt.show()
        # plt.plot(x_y_random[:, 0], x_y_random[:, 1], marker='.', color='k', linestyle='none')
        # plt.show()
        scene_start_id[scene] = sample
        for i in range(x_y_random.shape[0]):
            theta = np.arctan2(x_y_random[i, 1], x_y_random[i, 0])
            theta = min_max_scale(theta, range=[-np.pi, np.pi], target_range=[-1, 1])  # it seems silly just to leave the rest the same
            rad = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            state = clutter.RealState(obs_dict=data[scene], angle=-rad, sort=True, normalize=True, spherical=False,
                              translate_wrt_target=True)

            r = np.linalg.norm(x_y_random[i])
            if r > 1:
                r = 1
            push = clutter.PushTargetRealWithObstacleAvoidance(data[scene], theta=theta, push_distance=-1, distance=r,
                                                       push_distance_range=params['env']['params']['push']['distance'],
                                                       init_distance_range=params['env']['params']['push']['target_init_distance'],
                                                       translate_wrt_target=False)
            p = push.get_init_pos()

            # from mpl_toolkits.mplot3d import Axes3D
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.plot([p[0]], [p[1]], [0], color=[0, 0, 0], marker='o')
            # state.plot(ax=ax)
            # plt.show()

            dataset_x[sample] = np.append(state.array(), r)
            rewardd = reward(data[scene], p, r)
            dataset_y[sample] = rewardd
            sample += 1
    print(scene_start_id)

    with open(os.path.join(dir, 'dataset' + str(n_x) + 'x' + str(n_y) +  '.pkl'), 'wb') as file:
        pickle.dump([dataset_x, dataset_y, scene_start_id], file)

def train_supervised_critic(dir, dataset_name):
    rb_logging.init(directory=dir, friendly_name='', file_level=logging.INFO)
    with open(os.path.join(dir, dataset_name), 'rb') as file:
        dataset = pickle.load(file)
    agent = Critic()
    trainer = SupervisedTrainWorld(agent, dataset, epochs=150, save_every=10)
    trainer.run()
    print('Logging dir:', trainer.log_dir)

def visualize_supervised_output(model_dir, scenes_dir):
    from robamine.envs.clutter_utils import predict_collision
    def reward(obs_dict, p, distance):
        if predict_collision(obs_dict, p[0], p[1]):
            return -1

        reward = 1 - min_max_scale(distance, range=[-1, 1], target_range=[0, 1])
        return reward
    critic = Critic.load(os.path.join(model_dir, 'model_20.pkl'))
    density = 32
    scene_id = 12

    with open(os.path.join(scenes_dir, 'scenes.pkl'), 'rb') as file:
        data, _ = pickle.load(file)

    scenes_to_plot = len(data)
    for scene_id in range(scenes_to_plot):
        print('Processing scene', scene_id, 'out of', scenes_to_plot)


        x = np.linspace(-0.1, .1, density)
        y = np.linspace(-.1, .1, density)
        sizee = len(x)
        x, y = np.meshgrid(x, y)
        fig, axs = plt.subplots()
        z = np.zeros((sizee, sizee))
        for i in range(sizee):
            for j in range(sizee):
                theta = np.arctan2(y[i, j], x[i, j])
                rad = theta
                state = clutter.RealState(obs_dict=data[scene_id], angle=-rad, sort=True, normalize=True, spherical=False,
                                  translate_wrt_target=True)


                r = np.sqrt((10*x[i, j]) ** 2 + (10*y[i, j]) ** 2)
                if r > 1:
                    z[i, j] = 0
                else:
                    r = min_max_scale(r, range=[0, 1], target_range=[-1, 1])
                    push = clutter.PushTargetRealWithObstacleAvoidance(data[scene_id], theta=theta, push_distance=-1, distance=r,
                                                               push_distance_range=params['env']['params']['push'][
                                                                   'distance'],
                                                               init_distance_range=params['env']['params']['push'][
                                                                   'target_init_distance'],
                                                               translate_wrt_target=False)
                    p = push.get_init_pos()
                    z[i, j] = reward(data[scene_id], np.array([x[i, j], y[i, j]]) + data[scene_id]['object_poses'][0, 0:2], r)
                    # z[i, j] = critic.predict(np.append(state.array(), r))

        co = axs.contourf(x, y, z)
        fig.colorbar(co, ax=axs)
        plt.savefig(os.path.join(scenes_dir, 'screenshots/' + str(scene_id) + '_prediction.png'))
        plt.close()


# Obstacle avoidance
# ------------------

def test_obstacle_avoidance_critic(params):
    import matplotlib.pyplot as plt
    from robamine.envs.clutter_utils import get_table_point_cloud

    params['env']['params']['render'] = True
    params['env']['params']['safe'] = False
    env = gym.make(params['env']['name'], params=params['env']['params'])
    seed = np.random.randint(100000000)
    print('Seed:', seed)
    obs = env.reset(seed=seed)

    poses = obs['object_poses'][obs['object_above_table']]
    poses[:, :3] -= obs['object_poses'][obs['object_above_table']][0, :3]


    workspace = [.25, .25]
    table_point_cloud = get_table_point_cloud(poses, obs['object_bounding_box'][obs['object_above_table']], workspace)

    fig, ax = plt.subplots()
    ax.scatter(table_point_cloud[:, 0], table_point_cloud[:, 1], marker='o')
    plt.show()

    density = 64
    x = np.linspace(-.35, .35, density)
    y = np.linspace(-.35, .35, density)
    sizee = len(x)
    x, y = np.meshgrid(x, y)
    fig, axs = plt.subplots()
    z = np.zeros((sizee, sizee))
    max_min_dist = 0.1
    for i in range(sizee):
        for j in range(sizee):
            k = np.array([x[i, j], y[i, j]])
            diff = k - table_point_cloud
            min_dist = min(np.min(np.linalg.norm(diff, axis=1)), max_min_dist)
            if min_dist > 0.002 and abs(k[0]) <= workspace[0] and abs(k[1]) <= workspace[1]:
                z[i, j] = - min_max_scale(min_dist, range=[0, max_min_dist], target_range=[0, 1])
            else:
                z[i, j] = 0

    co = axs.contourf(x, y, z)
    fig.colorbar(co, ax=axs)
    plt.show()

def test_obstacle_avoidance_critic_torch(params):
    import matplotlib.pyplot as plt
    from robamine.algo.splitddpg import obstacle_avoidance_critic
    import torch


    params['env']['params']['render'] = True
    params['env']['params']['safe'] = False
    env = gym.make(params['env']['name'], params=params['env']['params'])
    seed = np.random.randint(100000000)
    print('Seed:', seed)
    obs = env.reset(seed=seed)
    poses = obs['object_poses'][obs['object_above_table']]
    bbox = obs['object_bounding_box'][obs['object_above_table']]

    torch.manual_seed(seed)
    batch_action = torch.FloatTensor(np.zeros((4096, 2))).uniform_(-1, 1)
    print('batch_action', batch_action)
    distance_range = [0, 0.1]

    q_value = obstacle_avoidance_critic(batch_action, distance_range, poses, bbox, bbox_aug=0.008, density=128)
    print('q value', q_value)


# Eval clutter in random policy
# -----------------------------

class RandomPolicy(RLAgent):
    def __init__(self):
        super(RandomPolicy, self).__init__(state_dim=None, action_dim=None, name='RandomPolicy')
        self.params = {}
        self.rng = np.random.RandomState()
        self.actions = [2, 1]

    def explore(self, state):
        i = self.rng.randint(0, len(self.actions))
        # i = 0
        output = np.zeros(self.actions[i] + 1)
        action = self.rng.uniform(-1, 1, self.actions[i])
        output[0] = i
        output[1:(self.actions[i] + 1)] = action
        return output

    def predict(self, state):
        return self.explore(state)

    def learn(self, transition):
        pass

    def q_value(self, state, action):
        return 0.0

    def seed(self, seed):
        self.rng.seed(seed)

    def save(self, path):
        pass

class RandomICRAPolicy(RLAgent):
    def __init__(self):
        super(RandomICRAPolicy, self).__init__(state_dim=None, action_dim=None, name='RandomICRAPolicy')
        self.params = {}
        self.rng = np.random.RandomState()

    def explore(self, state):
        return self.rng.randint(0, 3 * 8)

    def predict(self, state):
        return self.explore(state)

    def learn(self, transition):
        pass

    def q_value(self, state, action):
        return 0.0

    def seed(self, seed):
        self.rng.seed(seed)

    def save(self, path):
        pass


def eval_random_actions(params, n_scenes=1000):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='', file_level=logging.INFO)

    params['env']['params']['render'] = False
    params['env']['params']['deterministic_policy'] = True
    params['env']['params']['safe'] = True
    params['env']['params']['hardcoded_primitive'] = -1
    params['env']['params']['log_dir'] = params['world']['logging_dir']
    params['world']['episodes'] = n_scenes

    policy = RandomPolicy()
    world = EvalWorld(agent=policy, env=params['env'], params=params['world'])
    world.seed(0)
    world.run()
    print('Logging dir:', params['world']['logging_dir'])


# Train VAE for visual observation dimensionality reduction
# ---------------------------------------------------------


def VAE_collect_scenes(params, dir_to_save, n_scenes=1000):
    print('Collecting heightmaps of scenes...')
    from robamine.envs.clutter_cont import ClutterContWrapper
    from robamine.envs.clutter_utils import InvalidEnvError
    import copy

    def safe_reset(env):
        seed = np.random.randint(1000000000)
        try:
            obs = env.reset(seed=seed)
        except InvalidEnvError as e:
            print("WARN: {0}. Invalid environment during reset. A new environment will be spawn.".format(e))
            return safe_reset(env)
        return obs, seed

    params['env']['params']['render'] = False
    params['env']['params']['push']['predict_collision'] = True
    params['env']['params']['push']['target_init_distance'][1] = 0.15
    params['env']['params']['safe'] = False
    params['env']['params']['target']['randomize_pos'] = False
    params['env']['params']['nr_of_obstacles'] = [1, 8]
    env = ClutterContWrapper(params=params['env']['params'])

    real_states = []
    keywords = ['heightmap_mask']

    screenshots_path = os.path.join(dir_to_save, 'screenshots')
    if not os.path.exists(screenshots_path):
        os.makedirs(screenshots_path)

    i = 0
    pickle_path = os.path.join(dir_to_save, 'scenes.pkl')
    assert not os.path.exists(pickle_path), 'File already exists in ' + pickle_path
    file = open(pickle_path, 'wb')
    pickle.dump(n_scenes, file)
    pickle.dump(params, file)
    while i < n_scenes:
        print('Collecting scenes. Scene: ', i, 'out of', n_scenes)
        scene = {}
        obs, seed = safe_reset(env)
        scene['seed'] = seed
        for key in keywords:
            scene[key] = copy.deepcopy(obs[key])
        pickle.dump(scene, file)
        plt.imsave(os.path.join(screenshots_path, str(i) + '_screenshot.png'), env.env.bgr)
        i += 1

        print('Collecting scenes. Scene: ', i, 'out of', n_scenes)
        action = np.array([0, np.random.uniform(-1, 1), 1, 1])
        collision = True
        tries = 0
        failed = False
        while collision:
            try:
                obs, _, done, info = env.step(action)
            except InvalidEnvError as e:
                failed = True
                break
            tries += 1
            collision = info['collision']
            if tries > 10:
                failed = True
                break

        if not failed:
            scene = {}
            scene['seed'] = seed
            for key in keywords:
                scene[key] = copy.deepcopy(obs[key])
            pickle.dump(scene, file)
            plt.imsave(os.path.join(dir_to_save, 'screenshots/' + str(i) + '_screenshot.png'), env.env.bgr)
            i += 1
        else:
            print('Collecting scenes. Scene: ', i, 'failed, producing new')

    file.close()


def VAE_create_dataset(dir, rotations=0):
    from robamine.utils.cv_tools import Feature
    from robamine.envs.clutter_utils import get_actor_visual_feature
    import h5py

    # def get_visual_feature(heightmap, mask, target_bounding_box_z, finger_height, angle=0, plot=False):
    #     thresholded = np.zeros(heightmap.array().shape)
    #     threshold = target_bounding_box_z - 1.5 * finger_height
    #     if threshold < 0:
    #         threshold = 0
    #     thresholded[heightmap.array() > threshold] = 1
    #     thresholded[mask.array() > 0] = 0.5
    #     visual_feature = Feature(thresholded).rotate(angle)
    #     visual_feature = visual_feature.crop(crop_area[0], crop_area[1])
    #     visual_feature = visual_feature.pooling(kernel=[2, 2], stride=2, mode='AVG')
    #     if plot:
    #         visual_feature.plot()
    #     return feature

    scenes_path = os.path.join(dir, 'scenes.pkl')
    file = open(scenes_path, 'rb')
    n_scenes = pickle.load(file)
    params = pickle.load(file)

    dataset_path = os.path.join(dir, 'dataset' + '.hdf5')
    file_ = h5py.File(dataset_path, "a")
    n_datapoints = n_scenes * rotations
    visual_features = file_.create_dataset('features', (n_datapoints, 128, 128), dtype='f')

    n_sampler = 0
    for i in range(n_scenes):
        scene = pickle.load(file)
        for i in range(rotations):
            theta = (360 / rotations) * i
            feature = get_actor_visual_feature(heightmap=scene['heightmap_mask'][0],
                                               mask=scene['heightmap_mask'][1],
                                               target_bounding_box_z=np.array(
                                                   [scene['heightmap_mask'][0][198, 198] / 2.0]),
                                               finger_height=0.005,
                                               angle=theta)
            visual_features[n_sampler] = feature.copy()
            n_sampler += 1

    file.close()

def VAE_visual_eval_in_scenes(ae_dir, n_scenes=50):
    """
    Runs a number of scenes and saves the plots of input and output of AE in results_dir
    """

    # Load autoencoder
    import robamine.algo.conv_vae as ae
    ae_path = os.path.join(ae_dir, 'model.pkl')
    with open(ae_path, 'rb') as file:
        model = torch.load(file, map_location='cpu')
    # print('model', model, ae_path)
    latent_dim = model['encoder.fc.weight'].shape[0]
    vae = ae.ConvVae(latent_dim).to('cpu')
    vae.load_state_dict(model)

    params['env']['params']['render'] = False
    params['env']['params']['nr_of_obstacles'] = [1, 8]
    env = gym.make(params['env']['name'], params=params['env']['params'])

    results_path = os.path.join(ae_dir, 'results_1_8')

    n_scenes_ = 0

    rng = np.random.RandomState()
    rng.seed(0)
    seeds = rng.randint(0, 99999999, n_scenes)
    print(seeds)

    prediction_errors = []
    while n_scenes_ < n_scenes:
        seed = int(seeds[n_scenes_])
        obs = env.reset(seed=seed)

        heightmap = obs['heightmap_mask'][0]
        mask = obs['heightmap_mask'][1]
        target_bounding_box_z = obs['target_bounding_box'][2]
        finger_height = obs['finger_height']
        visual_feature = clutter.get_actor_visual_feature(heightmap, mask, target_bounding_box_z, finger_height, 0,
                                                  0, plot=False)
        visual_feature = torch.FloatTensor(visual_feature).reshape(1, 1, visual_feature.shape[0],
                                                                   visual_feature.shape[1]).to('cpu')
        ae_output = vae(visual_feature).detach().cpu().numpy()[0, 0, :, :]
        visual_feature = visual_feature.detach().cpu().numpy()[0, 0, :, :]
        prediction_error = ae_output - visual_feature
        prediction_errors.append(np.linalg.norm(prediction_error))
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(visual_feature, cmap='gray', vmin=np.min(visual_feature), vmax=np.max(visual_feature))
        ax[0].set_title('Input')
        ax[1].imshow(ae_output, cmap='gray', vmin=np.min(ae_output), vmax=np.max(ae_output))
        ax[1].set_title('Output')
        ax[2].imshow(prediction_error, cmap='gray', vmin=np.min(prediction_error), vmax=np.max(prediction_error))
        ax[2].set_title('Diff')
        fig.suptitle('AE performance. Norm(Diff) = ' + str(np.linalg.norm(prediction_error)))
        plt.savefig(os.path.join(results_path, 'scene_' + str(n_scenes_) + '.png'), dpi=300)
        plt.close()
        n_scenes_ += 1

        while n_scenes_ < n_scenes:
            action = rng.uniform(-1, 1, 4)
            action[0] = 0
            obs, reward, done, info = env.step(action)
            # print('action', action, 'reward: ', reward, 'done:', done)
            if done:
                break

            heightmap = obs['heightmap_mask'][0]
            mask = obs['heightmap_mask'][1]
            target_bounding_box_z = obs['target_bounding_box'][2]
            finger_height = obs['finger_height']
            visual_feature = clutter.get_actor_visual_feature(heightmap, mask, target_bounding_box_z, finger_height, 0,
                                                              0, plot=False)
            visual_feature = torch.FloatTensor(visual_feature).reshape(1, 1, visual_feature.shape[0],
                                                                       visual_feature.shape[1]).to('cpu')
            ae_output = vae(visual_feature).detach().cpu().numpy()[0, 0, :, :]
            visual_feature = visual_feature.detach().cpu().numpy()[0, 0, :, :]
            prediction_error = ae_output - visual_feature
            prediction_errors.append(np.linalg.norm(prediction_error))
            print('norm prediction error pixel', np.linalg.norm(prediction_error))
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(visual_feature, cmap='gray', vmin=np.min(visual_feature), vmax=np.max(visual_feature))
            ax[0].set_title('Input')
            ax[1].imshow(ae_output, cmap='gray', vmin=np.min(ae_output), vmax=np.max(ae_output))
            ax[1].set_title('Output')
            ax[2].imshow(prediction_error, cmap='gray', vmin=np.min(prediction_error), vmax=np.max(prediction_error))
            ax[2].set_title('Diff')
            fig.suptitle('AE performance. Norm(Diff) = ' + str(np.linalg.norm(prediction_error)))
            plt.savefig(os.path.join(results_path, 'scene_' + str(n_scenes_) + '.png'), dpi=300)
            plt.close()
            n_scenes_ += 1

            print('n_scenes = ', n_scenes_, '/', n_scenes)

    print('mean prediction error:', np.mean(prediction_errors))
    print('max prediction error:', np.max(prediction_errors))
    print('min prediction error:', np.min(prediction_errors))

# Train ICRA
# ----------

def icra_check_transition(params):
    from robamine.envs.clutter_cont import ClutterContICRAWrapper
    params['env']['params']['render'] = True
    params['env']['params']['push']['predict_collision'] = False
    params['env']['params']['safe'] = False
    params['env']['params']['icra']['use'] = True
    env = ClutterContICRAWrapper(params['env']['params'])

    while True:
        seed = np.random.randint(100000000)
        # seed = 36263774
        # seed = 48114142
        # seed = 86177553
        print('Seed:', seed)
        rng = np.random.RandomState()
        rng.seed(seed)
        obs = env.reset(seed=seed)

        while True:
            for action in np.arange(0, 24, 1):
                # action = 8
                print('icra discrete action', action)
                obs, reward, done, info = env.step(action)
                print('reward: ', reward, 'done:', done)
            if done:
                break

def train_icra(params):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='', file_level=logging.INFO)
    from robamine.algo.splitdqn import SplitDQN

    from robamine.envs.clutter_cont import ClutterContICRAWrapper
    params['env']['params']['render'] = False
    params['env']['params']['safe'] = True
    params['env']['params']['icra']['use'] = True
    env = gym.make('ClutterContICRAWrapper-v0', params=params['env']['params'])

    agent = SplitDQN(state_dim=264 * 8, action_dim=8 * 2,
                     params={'replay_buffer_size': 1e6,
                             'batch_size': [64, 64],
                             'discount': 0.9,
                             'epsilon_start': 0.9,
                             'epsilon_end': 0.05,
                             'epsilon_decay': 20000,
                             'learning_rate': [1e-3, 1e-3],
                             'tau': 0.999,
                             'double_dqn': True,
                             'hidden_units': [[100, 100], [100, 100]],
                             'loss': ['mse', 'mse'],
                             'device': 'cpu',
                             'load_nets': '',
                             'load_buffers': '',
                             'update_iter': [1, 1, 5]
                            })
    trainer = TrainWorld(agent=agent, env=env, params=params['world']['params'])
    trainer.run()

def train_eval_icra(params):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='', file_level=logging.INFO)
    from robamine.algo.splitdqn import SplitDQN

    from robamine.envs.clutter_cont import ClutterContICRAWrapper
    params['env']['params']['render'] = False
    params['env']['params']['safe'] = True
    params['env']['params']['icra']['use'] = True
    env = gym.make('ClutterContICRAWrapper-v0', params=params['env']['params'])

    agent = SplitDQN(state_dim=263 * 8, action_dim=8 * 3,
                     params={'replay_buffer_size': 1e6,
                             'batch_size': [64, 64, 64],
                             'discount': 0.9,
                             'epsilon_start': 0.9,
                             'epsilon_end': 0.25,
                             'epsilon_decay': 20000,
                             'learning_rate': [1e-3, 1e-3, 1e-3],
                             'tau': 0.999,
                             'double_dqn': True,
                             'hidden_units': [[100, 100], [100, 100], [100, 100]],
                             'loss': ['mse', 'mse', 'mse'],
                             'device': 'cpu',
                             'load_nets': '',
                             'load_buffers': '',
                             'update_iter': [1, 1, 1],
                             'n_preloaded_buffer': [500, 500, 500]
                             })
    trainer = TrainEvalWorld(agent=agent, env=env,
                             params={'episodes': 10000,
                                     'eval_episodes': 20,
                                     'eval_every': 100,
                                     'eval_render': False,
                                     'save_every': 100})
    trainer.seed(0)
    trainer.run()

def eval_random_actions_icra(params, n_scenes=1000):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='', file_level=logging.INFO)

    params['env']['params']['render'] = False
    params['env']['params']['safe'] = True
    params['env']['params']['hardcoded_primitive'] = -1
    params['env']['params']['log_dir'] = params['world']['logging_dir']
    params['world']['episodes'] = n_scenes
    env = gym.make('ClutterContICRAWrapper-v0', params=params['env']['params'])

    policy = RandomICRAPolicy()
    world = EvalWorld(agent=policy, env=env, params=params['world'])
    world.seed(0)
    world.run()
    print('Logging dir:', world.log_dir)

if __name__ == '__main__':
    hostname = socket.gethostname()
    exp_dir = 'robamine_logs_dream_2020.07.02.18.35.45.636114'

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

    # Basic runs
    # ----------

    # train(params)
    # train_eval(params)
    # train_combo_q_learning(params)
    # eval_with_render(os.path.join(params['world']['logging_dir'], exp_dir))
    # eval_in_scenes(params, os.path.join(params['world']['logging_dir'], exp_dir), n_scenes=1000)
    # analyze_eval_in_scenes(os.path.join(params['world']['logging_dir'], exp_dir))
    # exps = ['../ral-results/env-hard/random-discrete', '../ral-results/env-hard/random-cont',
    #         '../ral-results/env-hard/splitac-scratch/eval', '../ral-results/env-hard/splitdqn/eval',
    #         '../ral-results/env-icra/splitdqn-3/eval', '../ral-results/env-icra/splitac-scratch/eval']
    # names = ['Random-Discrete', 'Random-Cont', 'SplitAC-scratch', 'SplitDQN', 'SplitDQN@Env-ICRA', 'SplitAC-scr@Env-ICRA']
    # analyze_multiple_evals([os.path.join(params['world']['logging_dir'], _) for _ in exps], names)
    # analyze_multiple_eval_envs(params['world']['logging_dir'],
    #                            results_dir=os.path.join(logging_dir, '../ral-results/results'))
    # process_episodes(os.path.join(params['world']['logging_dir'], exp_dir))
    # check_transition(params)
    # test(params)
    # visualize_critic_predictions_2d(os.path.join(params['world']['logging_dir'], exp_dir))
    # visualize_critic_predictions(os.path.join(params['world']['logging_dir'], exp_dir))

    # Supervised learning
    # -------------------

    # collect_scenes_real_state(params, os.path.join(logging_dir, 'supervised_scenes/testing'), n_scenes=30)
    # create_dataset_from_scenes(os.path.join(logging_dir, 'supervised_scenes/training'), 16, 16)
    # create_dataset_from_scenes(os.path.join(logging_dir, 'supervised_scenes/training'), 16, 16)
    # create_dataset_from_scenes(os.path.join(logging_dir, 'supervised_scenes/training'), 32, 32)
    # train_supervised_critic(os.path.join(logging_dir, 'supervised_scenes/training'), 'dataset8x8.pkl')
    # train_supervised_critic(os.path.join(logging_dir, 'supervised_scenes/training'), 'dataset16x16.pkl')
    # train_supervised_critic(os.path.join(logging_dir, 'supervised_scenes/training'), 'dataset32x32.pkl')
    # visualize_supervised_output(model_dir=os.path.join(logging_dir, 'supervised_scenes/training/robamine_logs_triss_2020.05.24.15.32.16.980679'),
    #                             scenes_dir=os.path.join(logging_dir, 'supervised_scenes/testing'))

    # eval_random_actions(params, n_scenes=1000)


    # VAE training
    # ------------

    # VAE_path = os.path.join(logging_dir, 'VAE')
    # VAE_collect_scenes(params, dir_to_save=VAE_path, n_scenes=5000)
    # VAE_create_dataset(dir=VAE_path, rotations=16)
    # import robamine.algo.conv_vae as ae
    # ae.train(dir=VAE_path, split_per=0.9)
    # ae.test_vae(dir=VAE_path, model_epoch=60, split_per=0.9)
    # ae.estimate_normalizer(dir=VAE_path, model_epoch=60, split_per=0.9)
    # VAE_visual_eval_in_scenes(ae_dir=VAE_path)

    # ICRA comparison
    # ---------------

    # icra_check_transition(params)
    # train_icra(params)
    # train_eval_icra(params)
    # eval_random_actions_icra(params, n_scenes=1000)
