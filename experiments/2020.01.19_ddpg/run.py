from robamine.algo.core import TrainWorld, EvalWorld, SupervisedTrainWorld
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
from robamine.envs.clutter_utils import (plot_point_cloud_of_scene, discretize_2d_box, PushTargetFeature, RealState,
                                         preprocess_real_state, plot_real_state, PushTargetRealWithObstacleAvoidance)
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
    world.seed_list = np.arange(33, 40, 1).tolist()
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
    config['world']['episodes'] = n_scenes
    world = EvalWorld.load(dir, overwrite_config=config)
    world.seed_list = np.arange(0, n_scenes, 1).tolist()
    world.run()
    print('Logging dir:', params['world']['logging_dir'])

def process_episodes(dir):
    data = EpisodeListData.load(os.path.join(dir, 'episodes'))
    data.calc()
    print('Nr of episodes:', len(data))
    print('Success rate', data.success_rate)
    for episode in data:
        print(episode.actions_performed)

def check_transition(params):
    '''Run environment'''


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
        while True:
            action = rng.uniform(-1, 1, 4)
            action[0] = 0
            # action_ = np.zeros(action.shape)
            # action_[1] = min_max_scale(action[1], range=[-1, 1], target_range=[-pi, pi])
            # action_[2] = min_max_scale(action[2], range=[-1, 1], target_range=params['env']['params']['push']['distance'])
            # action_[3] = min_max_scale(action[3], range=[-1, 1], target_range=params['env']['params']['push']['target_init_distance'])
            # RealState(obs).plot(action=action_)
            # action = [0, -1, 1, -1]
            print('action', action)
            obs, reward, done, info = env.step(action)
            RealState(obs).plot()
            array = RealState(obs).array()
            if (array > 1).any() or (array < -1).any():
                print('out of limits indeces > 1:', array[array < -1])
                print('out of limits indeces < -1:', array[array < -1])
            # plot_point_cloud_of_scene(obs)
            print('reward: ', reward, 'done:', done)
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
            state = RealState(obs_dict=data[scene], angle=-rad, sort=True, normalize=True, spherical=False,
                              translate_wrt_target=True)

            r = np.linalg.norm(x_y_random[i])
            if r > 1:
                r = 1
            push = PushTargetRealWithObstacleAvoidance(data[scene], theta=theta, push_distance=-1, distance=r,
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
                state = RealState(obs_dict=data[scene_id], angle=-rad, sort=True, normalize=True, spherical=False,
                                  translate_wrt_target=True)


                r = np.sqrt((10*x[i, j]) ** 2 + (10*y[i, j]) ** 2)
                if r > 1:
                    z[i, j] = 0
                else:
                    r = min_max_scale(r, range=[0, 1], target_range=[-1, 1])
                    push = PushTargetRealWithObstacleAvoidance(data[scene_id], theta=theta, push_distance=-1, distance=r,
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
        self.params = params
        self.rng = np.random.RandomState()
        self.actions = [2]

    def explore(self, state):
        # i = self.rng.randint(0, len(self.actions))
        i = 0
        output = np.zeros(self.actions[i] + 1)
        action = self.rng.uniform(-1, 1, self.actions[0])
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

def eval_random_actions(params, n_scenes=1000):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='', file_level=logging.INFO)

    params['env']['params']['render'] = True
    params['env']['params']['safe'] = False
    params['env']['params']['hardcoded_primitive'] = 0
    params['env']['params']['log_dir'] = params['world']['logging_dir']
    params['world']['episodes'] = n_scenes

    policy = RandomPolicy()
    world = EvalWorld(agent=policy, env=params['env'], params=params['world'])
    world.seed_list = np.arange(9, n_scenes, 1).tolist()
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
    env = ClutterContWrapper(params=params['env']['params'])

    real_states = []
    keywords = ['heightmap_mask']

    i = 0
    while i < n_scenes:
        print('Collecting scenes. Scene: ', i, 'out of', n_scenes)
        scene = {}
        obs, seed = safe_reset(env)
        scene['seed'] = seed
        for key in keywords:
            scene[key] = copy.deepcopy(obs[key])
        real_states.append(scene)
        plt.imsave(os.path.join(dir_to_save, 'screenshots/' + str(i) + '_screenshot.png'), env.env.bgr)
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
            real_states.append(scene)
            plt.imsave(os.path.join(dir_to_save, 'screenshots/' + str(i) + '_screenshot.png'), env.env.bgr)
            i += 1
        else:
            print('Collecting scenes. Scene: ', i, 'failed, producing new')

    with open(os.path.join(dir_to_save, 'scenes.pkl'), 'wb') as file:
        pickle.dump([real_states, params], file)


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

    scenes, params = pickle.load(open(dir + 'scenes.pkl', 'rb'))
    n_scenes = len(scenes)

    file_ = h5py.File(os.path.join(dir, 'dataset' + '.hdf5'), "a")
    n_datapoints = n_scenes * rotations
    visual_features = file_.create_dataset('features', (n_datapoints, 128, 128), dtype='f')

    n_sampler = 0
    for scene in scenes:
        for i in range(rotations):
            theta = (360 / rotations) * i
            feature = get_actor_visual_feature(heightmap=scene['heightmap_mask'][0],
                                               mask=scene['heightmap_mask'][1],
                                               target_bounding_box_z=np.array(
                                                   [scene['heightmap_mask'][0][198, 198] / 2.0]),
                                               finger_height=0.005,
                                               angle=theta, plot=True)
            visual_features[n_sampler] = feature.copy()
            n_sampler += 1


if __name__ == '__main__':
    hostname = socket.gethostname()
    exp_dir = 'test'

    yml_name = 'params.yml'
    if hostname == 'dream':
        logging_dir = '/home/espa/robamine_logs/'
    elif hostname == 'triss':
        logging_dir = '/home/iason/robamine_logs/2020.01.16.split_ddpg'
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
    # eval_with_render(os.path.join(params['world']['logging_dir'], exp_dir))
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

    # eval_random_actions(params, n_scenes=10)

    # VAE training
    # ------------

    # VAE_collect_scenes(params,
    #                    dir_to_save='/home/mkiatos/robamine/logs/VAE',
    #                    n_scenes=1000)
    VAE_create_dataset(dir = '/home/mkiatos/robamine/logs/VAE/', rotations=16)
    # from robamine.algo.conv_vae import train, test_vae
    # train(dir = '/home/mkiatos/robamine/logs/VAE/')
    # test_vae(dir = '/home/mkiatos/robamine/logs/VAE/')
