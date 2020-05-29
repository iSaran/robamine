from robamine.algo.core import SupervisedTrainWorld
from robamine.clutter.real_mdp import RealState
from robamine.algo.core import Agent
from robamine import rb_logging
import logging
import yaml
import socket
import numpy as np
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from robamine.utils.math import min_max_scale
from robamine.utils.memory import get_batch_indices
from math import pi, floor, sqrt

from robamine.envs.clutter_utils import get_table_point_cloud

logger = logging.getLogger('robamine')
import h5py


# Supervised Obstacle Avoidance in loss of Actor
# ----------------------------------------------

class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(ActorNet, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_units[0]))
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            stdv = 1. / sqrt(self.hidden_layers[i].weight.size(1))
            self.hidden_layers[i].weight.data.uniform_(-stdv, stdv)
            self.hidden_layers[i].bias.data.uniform_(-stdv, stdv)

        self.out = nn.Linear(hidden_units[-1], 2)
        stdv = 1. / sqrt(self.out.weight.size(1))
        self.out.weight.data.uniform_(-stdv, stdv)
        self.out.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        out = torch.tanh(self.out(x))
        return out


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
        obstacle_avoidance_signal = - min_max_scale(min_dist, range=self.min_dist_range, target_range=[0.0, 1],
                                                    lib='torch', device=self.device)
        close_center_signal = 0.2 - min_max_scale(distance, range=self.distance_range, target_range=[0, 0.2], lib='torch',
                                                device=self.device)
        final_signal = close_center_signal
        final_signal[min_dist > self.min_dist_range[0]] += obstacle_avoidance_signal[min_dist > self.min_dist_range[0]]
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


class Actor(Agent):
    '''
    A class for train PyTorch networks. Inherit and create a self.network (which
    inherits from torch.nn.Module) before calling super().__init__()
    '''
    def __init__(self):
        super().__init__(name='Critic', params={})
        self.device = 'cpu'
        self.learning_rate = 0.001
        self.batch_size = 32
        # self.hidden_units = [800, 500, 300]
        self.hidden_units = [400, 300]

        self.network = ActorNet(RealState.dim(), self.hidden_units)

        # Create the networks, optimizers and loss
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=self.learning_rate)

        self.loss = ObstacleAvoidanceLoss(distance_range=[0, 0.1], min_dist_range=[0.002, 0.1], device=self.device)

        self.iterations = 0
        self.info['train'] = {'loss': 0.0}
        self.info['test'] = {'loss': 0.0}

        self.train_state = None
        self.train_point_clouds = None
        self.test_state = None
        self.test_point_clouds = None
        self.dataset = None

    def load_dataset(self, dataset, split=0.8):
        self.dataset = dataset

        scenes = floor(split * len(self.dataset['states']))
        self.train_state = self.dataset['states'][:scenes, :]
        self.train_point_clouds = self.dataset['point_clouds'][:scenes, :]
        self.test_state = self.dataset['states'][scenes:, :]
        self.test_point_clouds = self.dataset['point_clouds'][scenes:, :]

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

    def calc_losses(self, in_batches=False, batch_size=1000):
        if in_batches:
            # Calculate loss in train dataset
            loss_b = get_batch_indices(dataset_size=self.train_state.shape[0], batch_size=batch_size)
            train_loss = 0.0
            for i in range(int(len(loss_b))):
                train_state = torch.FloatTensor(self.train_state[loss_b[i]]).to(self.device)
                prediction = self.network(train_state)
                point_clouds = torch.FloatTensor(self.train_point_clouds[loss_b[i]]).to(self.device)
                loss = self.loss(point_clouds, prediction)
                train_loss += loss.detach().cpu().numpy().copy()
            self.info['train']['loss'] = train_loss / len(loss_b)

            loss_b = get_batch_indices(dataset_size=self.test_state.shape[0], batch_size=batch_size)
            test_loss = 0.0
            for i in range(int(len(loss_b))):
                test_state = torch.FloatTensor(self.test_state[loss_b[i]]).to(self.device)
                prediction = self.network(test_state)
                point_clouds = torch.FloatTensor(self.test_point_clouds[loss_b[i]]).to(self.device)
                loss = self.loss(point_clouds, prediction)
                test_loss += loss.detach().cpu().numpy().copy()
            self.info['test']['loss'] = test_loss / len(loss_b)
        else:
            # Calculate loss in train dataset
            train_state = torch.FloatTensor(self.train_state).to(self.device)
            prediction = self.network(train_state)
            point_clouds = torch.FloatTensor(self.train_point_clouds).to(self.device)
            loss = self.loss(point_clouds, prediction)
            self.info['train']['loss'] = loss.detach().cpu().numpy().copy()

            test_state = torch.FloatTensor(self.test_state).to(self.device)
            prediction = self.network(test_state)
            point_clouds = torch.FloatTensor(self.test_point_clouds).to(self.device)
            loss = self.loss(point_clouds, prediction)
            self.info['test']['loss'] = loss.detach().cpu().numpy().copy()

    def learn(self):
        '''Run one epoch'''
        self.iterations += 1
        self.calc_losses(in_batches=False)

        # Minimbatch update of network
        minibatches = get_batch_indices(dataset_size=self.train_state.shape[0], batch_size=self.batch_size)
        for minibatch in minibatches:
            batch_state = torch.FloatTensor(self.train_state[minibatch]).to(self.device)
            batch_point_clouds = torch.FloatTensor(self.train_point_clouds[minibatch]).to(self.device)
            prediction = self.network(batch_state)
            loss = self.loss(point_clouds=batch_point_clouds, actions=prediction)
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


def collect_scenes(params, dir_to_save, n_scenes=1000):
    print('Collecting the real state of some scenes...')
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
    params['env']['params']['safe'] = False
    params['env']['params']['target']['randomize_pos'] = False
    env = ClutterContWrapper(params=params['env']['params'])

    real_states = []
    keywords = ['object_poses', 'object_above_table', 'object_bounding_box', 'max_n_objects', 'surface_size',
                'finger_height', 'n_objects']

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

    with open(os.path.join(dir_to_save, 'scenes.pkl'), 'wb') as file:
        pickle.dump([real_states, params], file)


def merge_collected_scenes(dir, names):
    with open(os.path.join(dir, names[0] + '/scenes.pkl'), 'rb') as file:
        params_ = pickle.load(file)[1]

    data = []
    for name in names:
        with open(os.path.join(dir, name + '/scenes.pkl'), 'rb') as file:
            d = pickle.load(file)[0]
            for scene in d:
                data.append(scene)

    with open(os.path.join(dir, 'merged_scenes.pkl'), 'wb') as file:
        pickle.dump([data, params_], file)


def create_dataset_from_scenes(dir, rotations_augmentation=1, density=128):
    from robamine.utils.orientation import transform_poses, Quaternion, rot_z
    from robamine.clutter.real_mdp import RealState
    import copy
    import h5py
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from robamine.utils.viz import plot_boxes, plot_frames
    with open(os.path.join(dir, 'scenes.pkl'), 'rb') as file:
        data, params = pickle.load(file)


    # x = np.linspace(-1, 1, n_x)
    # y = np.linspace(-1, 1, n_y)
    # x, y = np.meshgrid(x, y)
    # x_y = np.column_stack((x.ravel(), y.ravel()))
    # distance = np.linalg.norm(x_y[0, :] - x_y[1, :])
    # x_y = x_y[np.linalg.norm(x_y, axis=1) <= 1]
    #
    n_scenes = len(data)
    # n_datapoints = n_scenes * x_y.shape[0]
    # n_features = 125
    # dataset_x = np.zeros((n_datapoints, n_features))
    # dataset_y = np.zeros((n_datapoints, 1))
    #
    # sample = 0
    # scene_start_id = np.zeros(len(data), dtype=np.int32)

    angle = np.linspace(0, 2 * np.pi, rotations_augmentation, endpoint=False)
    dataset = []
    max_init_distance = params['env']['params']['push']['target_init_distance'][1]
    max_obs_bounding_box = np.max(params['env']['params']['obstacle']['max_bounding_box'])
    # print(max_init_distance)

    # Create h5py file for storing dataset
    file_ = h5py.File(os.path.join(dir, 'dataset' + '.hdf5'), "a")
    n_datapoints  = n_scenes * rotations_augmentation
    states = file_.create_dataset('states', (n_datapoints, RealState.dim()), dtype='f')
    point_clouds = file_.create_dataset('point_clouds', (n_datapoints, density ** 2, 2), dtype='f')
    sample_counter = 0
    for scene in range(n_scenes):
        print('Creating dataset scenes. Scene: ', scene, 'out of', n_scenes)
        scene += 1

        for j in range(rotations_augmentation):
            state = copy.deepcopy(data[scene])

            # Uncomment to plot
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # plot_boxes(state['object_poses'][state['object_above_table']][:, 0:3],
            #            state['object_poses'][state['object_above_table']][:, 3:7],
            #            state['object_bounding_box'][state['object_above_table']], ax)
            # plot_frames(state['object_poses'][state['object_above_table']][:, 0:3],
            #             state['object_poses'][state['object_above_table']][:, 3:7], 0.01, ax)
            # ax.axis('equal')
            # plt.show()

            # Keep obstacles which are close to the object
            poses = state['object_poses'][state['object_above_table']]
            objects_close_target = np.linalg.norm(poses[:, 0:3] - poses[0, 0:3], axis=1) < (
                        max_init_distance + max_obs_bounding_box + 0.01)
            state['object_poses'] = state['object_poses'][state['object_above_table']][objects_close_target]
            state['object_bounding_box'] = state['object_bounding_box'][state['object_above_table']][objects_close_target]
            state['object_above_table'] = state['object_above_table'][state['object_above_table']][objects_close_target]

            # Keep obstacles around target and rotate state
            poses = state['object_poses'][state['object_above_table']]
            target_pose = poses[0].copy()
            poses = transform_poses(poses, target_pose)
            rotz = np.zeros(7)
            rotz[3:7] = Quaternion.from_rotation_matrix(rot_z(-angle[j])).as_vector()
            poses = transform_poses(poses, rotz)
            poses = transform_poses(poses, target_pose, target_inv=True)
            target_pose[3:] = np.zeros(4)
            target_pose[3] = 1
            poses = transform_poses(poses, target_pose)
            state['object_poses'][state['object_above_table']] = poses
            state['surface_angle'] = angle[j]

            real_state = RealState(state, angle=0, sort=True, normalize=True, spherical=False, range_norm=[-1, 1],
                                   translate_wrt_target=False)
            states[sample_counter] = real_state.array()

            # Uncomment to plot
            # print('rotated state:')
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # plot_boxes(state['object_poses'][state['object_above_table']][:, 0:3],
            #            state['object_poses'][state['object_above_table']][:, 3:7],
            #            state['object_bounding_box'][state['object_above_table']], ax)
            # plot_frames(state['object_poses'][state['object_above_table']][:, 0:3],
            #             state['object_poses'][state['object_above_table']][:, 3:7], 0.01, ax)
            # ax.axis('equal')
            # real_state.plot(ax=ax)
            # plt.show()

            point_cloud = max_init_distance * np.ones((density ** 2, 2))
            point_cloud_ = get_table_point_cloud(state['object_poses'][state['object_above_table']],
                                                state['object_bounding_box'][state['object_above_table']],
                                                workspace=[max_init_distance, max_init_distance],
                                                density=density)
            point_cloud[:point_cloud_.shape[0]] = point_cloud_
            point_clouds[sample_counter] = point_cloud
            # # Plot point cloud
            # print('shape:', point_cloud.shape)
            # fig, ax = plt.subplots()
            # ax.scatter(point_cloud[:, 0], point_cloud[:, 1])
            # plt.show()

            sample_counter += 1

    file_.close()


def train_supervised_actor(dir, dataset_name='dataset'):
    rb_logging.init(directory=dir, friendly_name='', file_level=logging.INFO)
    file_ = h5py.File(os.path.join(dir, dataset_name + '.hdf5'), "r")
    agent = Actor()
    trainer = SupervisedTrainWorld(agent, file_, epochs=150, save_every=10)
    trainer.run()
    print('Logging dir:', trainer.log_dir)


def visualize_actor_obs_output(model_dir, scenes_dir):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from robamine.utils.viz import plot_boxes, plot_frames

    actor = Actor.load(os.path.join(model_dir, 'model.pkl'))
    with open(os.path.join(scenes_dir, 'scenes.pkl'), 'rb') as file:
        data, par = pickle.load(file)

    max_init_distance = par['env']['params']['push']['target_init_distance'][1]
    scenes_to_plot = len(data)
    for scene_id in range(scenes_to_plot):
        print('Processing scene', scene_id, 'out of', scenes_to_plot)
#
        state = data[scene_id]
        state['surface_angle'] = 0
        real_state = RealState(data[scene_id], angle=0, sort=True, normalize=True, spherical=False, range_norm=[-1, 1],
                               translate_wrt_target=False).array()

        prediction = actor.predict(real_state)
        theta = min_max_scale(prediction[0, 0], range=[-1, 1], target_range=[-pi, pi])
        distance = min_max_scale(prediction[0, 1], range=[-1, 1], target_range=[0, 0.1])
        x = distance * np.cos(theta)
        y = distance * np.sin(theta)

        fig = plt.figure()
        ax = Axes3D(fig)
        plot_boxes(state['object_poses'][state['object_above_table']][:, 0:3],
                   state['object_poses'][state['object_above_table']][:, 3:7],
                   state['object_bounding_box'][state['object_above_table']], ax)
        plot_frames(state['object_poses'][state['object_above_table']][:, 0:3],
                    state['object_poses'][state['object_above_table']][:, 3:7], 0.01, ax)
        ax.scatter(x, y, 0, color=[0, 0, 0])
        ax.axis('equal')
        plt.show()


def plot_obs_avoidance(scenes_dir):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from robamine.utils.viz import plot_boxes, plot_frames
    with open(os.path.join(scenes_dir, 'scenes.pkl'), 'rb') as file:
        data, par = pickle.load(file)

    scenes_to_plot = len(data)
    for scene_id in range(scenes_to_plot):
        print('Processing scene', scene_id, 'out of', scenes_to_plot)
        state = data[scene_id]
        state['surface_angle'] = 0

        fig = plt.figure()
        ax = Axes3D(fig)
        plot_boxes(state['object_poses'][state['object_above_table']][:, 0:3],
                   state['object_poses'][state['object_above_table']][:, 3:7],
                   state['object_bounding_box'][state['object_above_table']], ax)
        plot_frames(state['object_poses'][state['object_above_table']][:, 0:3],
                    state['object_poses'][state['object_above_table']][:, 3:7], 0.01, ax)
        ax.axis('equal')

        d_range = par['env']['params']['push']['target_init_distance']
        loss = ObstacleAvoidanceLoss(distance_range=d_range)
        density = 128
        point_cloud = get_table_point_cloud(state['object_poses'][state['object_above_table']],
                                             state['object_bounding_box'][state['object_above_table']],
                                             workspace=[d_range[1], d_range[1]],
                                             density=density)
        loss.plot(point_cloud)

        plt.show()


if __name__ == '__main__':
    hostname = socket.gethostname()
    exp_dir = 'test'

    yml_name = 'params.yml'
    if hostname == 'dream':
        logging_dir = '/home/espa/robamine_logs/'
    elif hostname == 'triss':
        logging_dir = '/home/iason/robamine_logs/2020.01.16.split_ddpg'
    else:
        raise ValueError()
    with open(yml_name, 'r') as stream:
        params = yaml.safe_load(stream)

    params['world']['logging_dir'] = logging_dir

    # collect_scenes(params, os.path.join(logging_dir, 'supervised_obstacle_avoidance/scenes'), n_scenes=20)
    # create_dataset_from_scenes(os.path.join(logging_dir, 'supervised_obstacle_avoidance/scenes'),
    #                                          rotations_augmentation=8)
    # train_supervised_actor(os.path.join(logging_dir, 'supervised_obstacle_avoidance/scenes'))

    # visualize_actor_obs_output('/home/espa/robamine_logs/supervised_obstacle_avoidance/scenes/robamine_logs_dream_2020.05.29.01.49.17.553991',
    # os.path.join(logging_dir, 'supervised_obstacle_avoidance/scenes'))
    plot_obs_avoidance(os.path.join(logging_dir, 'supervised_obstacle_avoidance/scenes'))
