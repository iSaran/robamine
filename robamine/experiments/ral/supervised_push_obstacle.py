from robamine.algo.core import SupervisedTrainWorld
import robamine.algo.core as core
from robamine import rb_logging
import logging
import yaml
import socket
import numpy as np
import os
import gym
import pickle

import torch
import torch.nn as nn

import robamine.envs.clutter_utils as clutter
import matplotlib.pyplot as plt
from robamine.utils.math import min_max_scale

import robamine.algo.conv_vae as ae
import h5py
import copy
from robamine.envs.clutter_cont import ClutterContWrapper
import math



logger = logging.getLogger('robamine')

# ----------------------------------------------------
# Push Obstacle w/ supervised learning
# ----------------------------------------------------


class ObsDictPolicy:
    def predict(self, obs_dict):
        raise NotImplementedError

    def __call__(self, obs_dict):
        return self.predict(obs_dict)

class PushObstacleRealPolicy(ObsDictPolicy):
    def predict(self, state):
        n_objects = int(state['n_objects'])
        target_pose = state['object_poses'][0]
        target_bbox = state['object_bounding_box'][0]

        distances = 100 * np.ones((int(n_objects),))
        for i in range(1, n_objects):
            obstacle_pose = state['object_poses'][i]
            obstacle_bbox = state['object_bounding_box'][i]

            distances[i] = clutter.get_distance_of_two_bbox(target_pose, target_bbox, obstacle_pose, obstacle_bbox)

        closest_obstacles = np.argwhere(distances < 0.03).flatten()
        errors = 2 * np.pi * np.ones(n_objects)
        for object in closest_obstacles:
            object_height = clutter.get_object_height(state['object_poses'][object], state['object_bounding_box'][object])

            # Ignore obstacles below threshold for push obstacle
            threshold = 2 * state['object_bounding_box'][0][2] + 1.1 * state['finger_height']
            if object_height < threshold:
                continue

            direction = state['object_poses'][object][0:2] - target_pose[0:2]
            direction /= np.linalg.norm(direction)
            angle = np.arctan2(direction[1], direction[0])
            if angle < 0:
                angle += 2 * np.pi

            base_direction = - target_pose[0:2]
            base_direction /= np.linalg.norm(base_direction)
            base_angle = np.arctan2(base_direction[1], base_direction[0])
            if base_angle < 0:
                base_angle += 2 * np.pi

            base_angle -= np.pi / 4
            if base_angle < 0:
                base_angle += 2 * np.pi

            error_angle = angle - base_angle
            if error_angle < 0:
                error_angle += 2 * np.pi

            errors[object] = error_angle

        object = np.argmin(errors)
        direction = state['object_poses'][object][0:2] - target_pose[0:2]
        theta = np.arctan2(direction[1], direction[0])
        theta = min_max_scale(theta, range=[-np.pi, np.pi], target_range=[-1, 1])
        return np.array([1, theta])

class PushObstacleFeature:
    def __init__(self, vae_path):
        # Load autoencoder and scaler
        ae_path = os.path.join(vae_path, 'model.pkl')
        with open(ae_path, 'rb') as file1:
            model = torch.load(file1, map_location='cpu')
        latent_dim = model['encoder.fc.weight'].shape[0]
        ae_params = ae.params
        ae_params['device'] = 'cpu'
        self.autoencoder = ae.ConvVae(latent_dim, ae_params)
        self.autoencoder.load_state_dict(model)

        self.heightmap = None
        self.mask = None
        self.target_bounding_box_z = None
        self.finger_height = None
        self.surface_distances = None
        self.target_pos = None

    def __call__(self, obs_dict, angle=0):
        self.set_dict(obs_dict)
        return clutter.get_asymmetric_actor_feature(self.autoencoder, None, self.heightmap, self.mask, self.target_bounding_box_z, self.finger_height,
                                            self.target_pos, self.surface_distances, 0, 1, False)

    def set_dict(self, obs_dict):
        self.heightmap = obs_dict['heightmap_mask'][0].copy()
        self.mask = obs_dict['heightmap_mask'][1].copy()
        self.target_bounding_box_z = obs_dict['target_bounding_box'][2].copy()
        self.finger_height = obs_dict['finger_height'].copy()
        self.surface_distances = [obs_dict['surface_size'][0] - obs_dict['object_poses'][0, 0], \
                             obs_dict['surface_size'][0] + obs_dict['object_poses'][0, 0], \
                             obs_dict['surface_size'][1] - obs_dict['object_poses'][0, 1], \
                             obs_dict['surface_size'][1] + obs_dict['object_poses'][0, 1]]
        self.surface_distances = np.array([x / 0.5 for x in self.surface_distances])
        self.target_pos = obs_dict['object_poses'][0, 0:2].copy()


    def plot(self, obs_dict):
        self.set_dict(obs_dict)
        visual_feature = clutter.get_actor_visual_feature(self.heightmap, self.mask, self.target_bounding_box_z, self.finger_height, 0,
                                                  1, plot=False)
        visual_feature = torch.FloatTensor(visual_feature).reshape(1, 1, visual_feature.shape[0],
                                                                   visual_feature.shape[1]).to('cpu')
        ae_output = self.autoencoder(visual_feature).detach().cpu().numpy()[0, 0, :, :]
        visual_feature = visual_feature.detach().cpu().numpy()[0, 0, :, :]
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(visual_feature, cmap='gray', vmin=np.min(visual_feature), vmax=np.max(visual_feature))
        ax[1].imshow(ae_output, cmap='gray', vmin=np.min(ae_output), vmax=np.max(ae_output))
        return fig, ax


class CosLoss(nn.Module):
    def forward(self, x1, x2):
        return torch.mean(torch.ones(x1.shape) - torch.cos(torch.abs(x1 - x2)))


class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout=0.0):
        super(ActorNet, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_units[0]))
        # self.hidden_layers.append(nn.Dropout(p=dropout))
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            # stdv = 1. / math.sqrt(self.hidden_layers[i].weight.size(1))
            # stdv = 1e-3
            # self.hidden_layers[-1].weight.data.uniform_(-stdv, stdv)
            # self.hidden_layers[-1].bias.data.uniform_(-stdv, stdv)
            # self.hidden_layers.append(nn.Dropout(p=dropout))

        self.out = nn.Linear(hidden_units[-1], 1)
        # stdv = 1. / math.sqrt(self.out.weight.size(1))
        # stdv = 1e-3
        # self.out.weight.data.uniform_(-stdv, stdv)
        # self.out.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        out = self.out(x)
        return out


class Actor(core.NetworkModel):
    def __init__(self, params, inputs=None, outputs=None):
        self.hidden_units = params['hidden_units']
        inputs = ae.LATENT_DIM
        self.network = ActorNet(ae.LATENT_DIM, self.hidden_units)
        if params['loss'] == 'cos':
            self.loss = CosLoss()
            params['loss'] = 'custom'

        super(Actor, self).__init__(name='Actor', inputs=inputs, outputs=1, params=params)

    @classmethod
    def load_state_dict(cls, state_dict):
        self = cls(state_dict['params'], state_dict['inputs'],
                   state_dict['outputs'])
        self.load_trainable_dict(state_dict['trainable'])
        self.iterations = state_dict['iterations']
        self.scaler_x = state_dict['scaler_x']
        self.scaler_y = state_dict['scaler_y']
        self.loss = CosLoss()
        return self

    def predict(self, state):
        prediction = super(Actor, self).predict(state)
        prediction = float(prediction)
        prediction -= np.sign(prediction) * np.int(np.ceil((np.abs(prediction) - np.pi) / (2 * np.pi))) * 2 * np.pi
        prediction = min_max_scale(prediction, range=[-np.pi, np.pi], target_range=[-1, 1])
        return np.array([prediction])


class PushObstacleSupervisedExp:
    def __init__(self, params, log_dir, vae_path, actor_path, seed, file_type='pkl', partial_dataset=None):
        self.params = copy.deepcopy(params)
        self.params['env']['params']['render'] = False
        self.params['env']['params']['push']['predict_collision'] = True
        self.params['env']['params']['safe'] = True
        self.params['env']['params']['target']['randomize_pos'] = True
        self.params['env']['params']['nr_of_obstacles'] = [1, 10]
        self.params['env']['params']['hardcoded_primitive'] = -1
        self.params['env']['params']['all_equal_height_prob'] = 0.0
        self.params['env']['params']['obstacle']['pushable_threshold_coeff'] = 1.0
        self.params['env']['params']['target']['max_bounding_box'] = [0.03, 0.03, 0.015]

        self.log_dir = log_dir
        self.vae_path = vae_path
        self.actor_path = actor_path
        self.seed = seed
        self.file_type = file_type
        self.partial_dataset = partial_dataset

    def collect_samples(self, n_samples=5000, plot=False):
        print('Push obstacle supervised: Collecting heightmaps of scenes...')

        generator = np.random.RandomState()
        generator.seed(self.seed)
        seed_scenes = generator.randint(0, 1e7, n_samples)

        if plot:
            self.params['env']['params']['render'] = True
        env = ClutterContWrapper(params=self.params['env']['params'])

        keywords = ['heightmap_mask', 'target_bounding_box', 'finger_height', 'surface_edges', 'surface_size',
                    'object_poses']

        pickle_path = os.path.join(self.log_dir, 'scenes' + str(self.seed) + '.pkl')
        assert not os.path.exists(pickle_path), 'File already exists in ' + pickle_path
        file = open(pickle_path, 'wb')
        pickle.dump(n_samples, file)
        pickle.dump(params, file)
        pickle.dump(seed_scenes, file)

        policy = PushObstacleRealPolicy()

        scene = 1
        sample = 1
        push_obstacle_feature = PushObstacleFeature(self.vae_path)
        while sample <= n_samples:
            print('Collecting scenes. Sample: ', sample, 'out of', n_samples, '. Scene seed:', seed_scenes[scene])
            obs = env.reset(seed=int(seed_scenes[scene]))

            while True:
                feature = clutter.get_actor_visual_feature(obs['heightmap_mask'][0], obs['heightmap_mask'][1],
                                                           obs['target_bounding_box'][2], obs['finger_height'],
                                                           primitive=1, maskout_target=True)
                if np.argwhere(feature >= 1).size == 0:
                    scene += 1
                    break

                action = policy(obs)
                next_obs, reward, done, info = env.step(action)

                if plot:
                    fig, ax = push_obstacle_feature.plot(obs)
                    feature = push_obstacle_feature(obs)

                    angle_rad = min_max_scale(action[1], range=[-1, 1], target_range=[-np.pi, np.pi])
                    theta = feature[-1]
                    print('thetaaaaaaa', theta)
                    if theta < 0:
                        theta += 2 * np.pi
                    print('thetaaaaaaa', theta)
                    theta -= np.pi / 4
                    if theta < 0:
                        theta += 2 * np.pi
                    length = 40
                    xy = [length * np.cos(- theta), length * np.sin(- theta)]
                    table_center = np.zeros(2)
                    table_center[0] = xy[0]
                    table_center[1] = xy[1]

                    center_image = [64, 64]
                    length = 40
                    x = length * np.cos(- angle_rad)
                    y = length * np.sin(- angle_rad)
                    print('center image', center_image)
                    print('x, y', x, y)
                    print('table center', table_center)
                    ax[0].arrow(center_image[0], center_image[1], x, y, head_width=3, color=[1, 0, 0])
                    ax[0].arrow(center_image[0], center_image[1], table_center[0], table_center[1], head_width=3, color=[0.0, 0.0, 1.0])
                    ax[1].arrow(center_image[0], center_image[1], x, y, head_width=3, color=[1, 0, 0])
                    ax[1].arrow(center_image[0], center_image[1], table_center[0], table_center[1], head_width=3, color=[0.0, 0.0, 1.0])

                    plt.show()

                if reward >= -0.1:
                    state = {}
                    for key in keywords:
                        state[key] = copy.deepcopy(obs[key])
                    pickle.dump([state, action.copy()], file)
                    sample += 1

                obs = next_obs

                if reward <= -0.25 or reward > 0 or sample > n_samples:
                    scene += 1
                    break

        file.close()

    def create_dataset(self, rotations=8):
        from robamine.envs.clutter_utils import get_actor_visual_feature
        import copy

        scenes_path = os.path.join(self.log_dir, 'scenes' + str(self.seed) + '.pkl')
        scenes_file = open(scenes_path, 'rb')
        n_samples = pickle.load(scenes_file)
        params = pickle.load(scenes_file)
        seed_scenes = pickle.load(scenes_file)

        dataset_path = os.path.join(self.log_dir, 'dataset' + str(self.seed) + '.' + self.file_type)
        if self.file_type == 'hdf5':
            dataset_file = h5py.File(dataset_path, "a")
        elif self.file_type == 'pkl':
            dataset_file = open(dataset_path, 'wb')
        else:
            raise ValueError()

        n_datapoints = n_samples * rotations

        push_obstacle_feature = PushObstacleFeature(self.vae_path)

        if self.file_type == 'hdf5':
            visual_features = dataset_file.create_dataset('features', (n_datapoints, ae.LATENT_DIM), dtype='f')
            actions = dataset_file.create_dataset('action', (n_datapoints,), dtype='f')
        elif self.file_type == 'pkl':
            visual_features = np.zeros((n_datapoints, ae.LATENT_DIM))
            actions = np.zeros(n_datapoints)
        else:
            raise ValueError()

        n_sampler = 0
        for k in range(n_samples):
            print('Processing sample', k, 'of', n_samples)
            sample = pickle.load(scenes_file)
            scene = sample[0]
            action = copy.copy(sample[1][1])

            angle = np.linspace(0, 2 * np.pi, rotations, endpoint=False)
            for i in range(rotations):
                angle_pi = angle[i]
                if angle_pi > np.pi:
                    angle_pi -= 2 * np.pi
                feature = push_obstacle_feature(scene, angle_pi)
                visual_features[n_sampler] = feature.copy()

                angle_pi = min_max_scale(angle_pi, range=[-np.pi, np.pi], target_range=[-1, 1])
                action_ = action + angle_pi
                if action_ > 1:
                    action_ = -1 + abs(1 - action_)
                elif action_ < -1:
                    action_ = 1 - abs(-1 - action_)
                actions[n_sampler] = action_
                n_sampler += 1

        if self.file_type == 'pkl':
            pickle.dump([visual_features, actions], dataset_file)
            dataset_file.close()

        scenes_file.close()

    def merge_datasets(self, seeds):
        assert self.file_type == 'pkl', 'For now only pkl'

        dataset_path = os.path.join(self.log_dir, 'dataset.' + self.file_type)
        dataset_file = open(dataset_path, 'wb')

        dataset_path = os.path.join(self.log_dir, 'dataset' + str(seeds[0]) + '.' + self.file_type)
        with open(dataset_path, 'rb') as dataset_file_i:
            data = pickle.load(dataset_file_i)
        for i in range(1, len(seeds)):
            dataset_path = os.path.join(self.log_dir, 'dataset' + str(seeds[i]) + '.' + self.file_type)
            with open(dataset_path, 'rb') as dataset_file_i:
                data_i = pickle.load(dataset_file_i)
            data[0] = np.append(data[0], data_i[0], axis=0)
            data[1] = np.append(data[1], data_i[1])

        pickle.dump(data, dataset_file)
        dataset_file.close()

    def scale_outputs(self, range_=[-1, 1], target_range_=[-np.pi, np.pi]):
        assert self.file_type == 'pkl', 'For now only pkl'
        dataset_path = os.path.join(self.log_dir, 'dataset.' + self.file_type)
        dataset_file = open(dataset_path, 'rb')
        dataset = pickle.load(dataset_file)

        for i in range(len(dataset[1])):
            dataset[1][i] = min_max_scale(dataset[1][i], range=range_, target_range=target_range_)

        dataset_scaled_path = os.path.join(self.log_dir, 'dataset_scaled.' + self.file_type)
        dataset_scaled_file = open(dataset_scaled_path, 'wb')
        pickle.dump(dataset, dataset_scaled_file)
        dataset_file.close()
        dataset_scaled_file.close()

    def train(self, hyperparams, epochs=150, save_every=10, suffix=''):
        from robamine.algo.util import Dataset
        rb_logging.init(directory=self.log_dir, friendly_name='actor' + suffix, file_level=logging.INFO)
        dataset_path = os.path.join(self.log_dir, 'dataset.' + self.file_type)
        if self.file_type == 'hdf5':
            dataset = h5py.File(dataset_path, "r")
        elif self.file_type == 'pkl':
            with open(dataset_path, 'rb') as file_:
                data = pickle.load(file_)
                if self.partial_dataset is None:
                    end_dataset = len(data[0])
                else:
                    end_dataset = self.partial_dataset
                data[0] = data[0][:end_dataset]
                data[1] = data[1][:end_dataset]
            dataset = Dataset.from_array(data[0], data[1].reshape(-1, 1))

        agent = Actor(hyperparams)
        trainer = SupervisedTrainWorld(agent, dataset, epochs=epochs, save_every=save_every)
        trainer.run()
        print('Logging dir:', trainer.log_dir)

    def visualize_dataset(self):
        dataset_path = os.path.join(self.log_dir, 'dataset.' + self.file_type)
        with open(dataset_path, 'rb') as file_:
            data = pickle.load(file_)

        # Load autoencoder and scaler
        ae_path = os.path.join(self.vae_path, 'model.pkl')
        normalizer_path = os.path.join(self.vae_path, 'normalizer.pkl')
        with open(ae_path, 'rb') as file1:
            model = torch.load(file1, map_location='cpu')
        latent_dim = model['encoder.fc.weight'].shape[0]
        ae_params = ae.params
        ae_params['device'] = 'cpu'
        autoencoder = ae.ConvVae(latent_dim, ae_params)
        autoencoder.load_state_dict(model)
        with open(normalizer_path, 'rb') as file2:
            normalizer = pickle.load(file2)

        grid = (8, 8)
        total_grid = grid[0] * grid[1]

        if self.actor_path is not None:
            # Load actor
            actor = Actor.load(self.actor_path)

        indeces = np.arange(0, total_grid, 1).reshape(grid)

        center_image = [64, 64]

        while True:

            fig, ax = plt.subplots(grid[0], grid[1])
            fig_train, ax_train = plt.subplots(grid[0], grid[1])

            seed = input('Enter seed (q to quit): ')
            if seed == 'q':
                break
            seed = int(seed)
            generator = np.random.RandomState()
            generator.seed(seed)
            if self.partial_dataset is None:
                end_dataset = len(data[0])
            else:
                end_dataset = self.partial_dataset
            test_samples = generator.randint(end_dataset * 0.9, end_dataset, total_grid)
            train_samples = generator.randint(0, end_dataset * 0.9, total_grid)
            # test_samples = np.arange(seed * total_grid * 4, seed * total_grid * 4 + total_grid * 4, 4)
            # test_samples = np.arange(0, 64, 1)

            for i in range(grid[0]):
                for j in range(grid[1]):
                    sample_index = test_samples[indeces[i, j]]
                    latent = data[0][sample_index][:256]
                    angle_rad = data[1][sample_index]
                    theta = data[0][sample_index][-1]
                    # if theta < 0:
                    #     theta += 2 * np.pi
                    # theta -= np.pi / 4
                    # if theta < 0:
                    #     theta += 2 * np.pi
                    length = 40
                    xy = [length * np.cos(- theta), length * np.sin(- theta)]
                    table_center = np.zeros(2)
                    table_center[0] = xy[0]
                    table_center[1] = xy[1]

                    # angle_rad = min_max_scale(angle, [-1, 1], [-np.pi, np.pi])
                    # latent = normalizer.inverse_transform(latent)
                    latent = torch.FloatTensor(latent).reshape(1, -1).to('cpu')
                    reconstruction = autoencoder.decoder(latent).detach().cpu().numpy()[0, 0, :, :]
                    ax[i, j].imshow(reconstruction, cmap='gray', vmin=np.min(reconstruction),
                                    vmax=np.max(reconstruction))
                    length = 40
                    x = length * np.cos(- angle_rad)
                    y = length * np.sin(- angle_rad)
                    ax[i, j].arrow(center_image[0], center_image[1], x, y, head_width=3, color=[1, 0, 0])
                    # ax[i, j].arrow(center_image[0], center_image[1], table_center[0], table_center[1], head_width=3, color=[0.0, 0.0, 1.0])

                    if self.actor_path is not None:
                        latent = data[0][test_samples[indeces[i, j]]]
                        angle_rad = actor.predict(latent)[0]
                        angle_rad = min_max_scale(angle_rad, [-1, 1], [-np.pi, np.pi])
                        x = length * np.cos(- angle_rad)
                        y = length * np.sin(- angle_rad)
                        ax[i, j].arrow(center_image[0], center_image[1], x, y, head_width=3, color=[0, 1, 0])
                    ax[i, j].set_title(str(sample_index))
            fig.suptitle('Test')

            for i in range(grid[0]):
                for j in range(grid[1]):
                    sample_index = train_samples[indeces[i, j]]
                    latent = data[0][sample_index][:256]
                    angle_rad = data[1][sample_index]
                    theta = data[0][sample_index][-1]
                    # if theta < 0:
                    #     theta += 2 * np.pi
                    # theta -= np.pi / 4
                    # if theta < 0:
                    #     theta += 2 * np.pi
                    length = 40
                    xy = [length * np.cos(- theta), length * np.sin(- theta)]
                    table_center = np.zeros(2)
                    table_center[0] = xy[0]
                    table_center[1] = xy[1]



                    # angle_rad = min_max_scale(angle, [-1, 1], [-np.pi, np.pi])
                    # latent = normalizer.inverse_transform(latent)
                    latent = torch.FloatTensor(latent).reshape(1, -1).to('cpu')
                    reconstruction = autoencoder.decoder(latent).detach().cpu().numpy()[0, 0, :, :]
                    ax_train[i, j].imshow(reconstruction, cmap='gray', vmin=np.min(reconstruction),
                                          vmax=np.max(reconstruction))
                    length = 40
                    x = length * np.cos(- angle_rad)
                    y = length * np.sin(- angle_rad)
                    ax_train[i, j].arrow(center_image[0], center_image[1], x, y, head_width=3, color=[1, 0, 0])
                    # ax_train[i, j].arrow(center_image[0], center_image[1], table_center[0], table_center[1], head_width=3, color=[0.0, 0.0, 1.0])

                    if self.actor_path is not None:
                        latent = data[0][train_samples[indeces[i, j]]]
                        angle_rad = actor.predict(latent)[0]
                        angle_rad = min_max_scale(angle_rad, [-1, 1], [-np.pi, np.pi])
                        x = length * np.cos(- angle_rad)
                        y = length * np.sin(- angle_rad)
                        ax_train[i, j].arrow(center_image[0], center_image[1], x, y, head_width=3, color=[0, 1, 0])
                    ax_train[i, j].set_title(str(sample_index))
            fig_train.suptitle('Train')
            plt.show()

    def visual_evaluation(self, plot=False):
        '''Run environment'''

        push_obstacle_feature = PushObstacleFeature(self.vae_path)
        actor_model = Actor.load(self.actor_path)

        params = copy.deepcopy(self.params)

        params['env']['params']['render'] = True
        params['env']['params']['deterministic_policy'] = True
        env = gym.make(params['env']['name'], params=params['env']['params'])

        while True:
            print('Seed:', self.seed)
            rng = np.random.RandomState()
            rng.seed(self.seed)
            obs = env.reset(seed=self.seed)

            while True:
                self.seed += 1
                feature = push_obstacle_feature(obs, angle=0)
                action = rng.uniform(-1, 1, 4)
                action[0] = 1
                angle = actor_model.predict(feature)[0]
                # print('angle predicted', angle)
                # angle += int(np.ceil((np.abs(angle) - np.pi) / (2 * np.pi))) * 2 * np.pi
                # print('angle predicted transformed', angle)
                # angle = min_max_scale(angle, range=[-np.pi, np.pi], target_range=[-1, 1])
                # print('angle predicted transformed 1, 1', angle)
                action[1] = angle

                if plot:
                    fig, ax = push_obstacle_feature.plot(obs)
                    feature = push_obstacle_feature(obs)

                    angle_rad = min_max_scale(action[1], range=[-1, 1], target_range=[-np.pi, np.pi])
                    theta = feature[-1]
                    length = 40
                    xy = [length * np.cos(- theta), length * np.sin(- theta)]
                    table_center = np.zeros(2)
                    table_center[0] = xy[0]
                    table_center[1] = xy[1]

                    center_image = [64, 64]
                    length = 40
                    x = length * np.cos(- angle_rad)
                    y = length * np.sin(- angle_rad)
                    ax[0].arrow(center_image[0], center_image[1], x, y, head_width=3, color=[1, 0, 0])
                    ax[0].arrow(center_image[0], center_image[1], table_center[0], table_center[1], head_width=3, color=[0.0, 0.0, 1.0])
                    ax[1].arrow(center_image[0], center_image[1], x, y, head_width=3, color=[1, 0, 0])
                    ax[1].arrow(center_image[0], center_image[1], table_center[0], table_center[1], head_width=3, color=[0.0, 0.0, 1.0])

                    plt.show()

                obs_next, reward, done, info = env.step(action)
                print('reward:', reward, 'action:', action, 'done:', done, 'temrination condiction:',
                      info['termination_reason'])
                # RealState(obs).plot()
                # array = RealState(obs).array()
                # if (array > 1).any() or (array < -1).any():
                #     print('out of limits indeces > 1:', array[array < -1])
                #     print('out of limits indeces < -1:', array[array < -1])
                # plot_point_cloud_of_scene(obs)
                if done:
                    break

                obs = obs_next

    def eval_in_scenes(self, n_scenes=1000, random_policy=False):
        '''Run environment'''

        push_obstacle_feature = PushObstacleFeature(self.vae_path)
        actor_model = Actor.load(self.actor_path)

        params = copy.deepcopy(self.params)
        params['env']['params']['render'] = False
        params['env']['params']['deterministic_policy'] = True
        env = gym.make(params['env']['name'], params=params['env']['params'])

        samples = 0
        successes = 0
        while True:
            print('Seed:', samples)
            rng = np.random.RandomState()
            rng.seed(samples)
            obs = env.reset(seed=samples)

            while True:
                samples += 1
                feature = push_obstacle_feature(obs, angle=0)
                action = rng.uniform(-1, 1, 4)
                action[0] = 1
                if random_policy:
                    angle = rng.uniform(-1, 1)
                else:
                    angle = actor_model.predict(feature)[0]

                # angle += int(np.ceil((np.abs(angle) - np.pi) / (2 * np.pi))) * 2 * np.pi
                # angle = min_max_scale(angle, range=[-np.pi, np.pi], target_range=[-1, 1])
                action[1] = angle
                obs_next, reward, done, info = env.step(action)
                if reward >= -0.1:
                    successes += 1

                print('reward:', reward, 'action:', action, 'done:', done, 'temrination condiction:',
                      info['termination_reason'])
                print('successes', successes, '/', samples, '(', (successes / samples) * 100, ') %')
                if done:
                    break

                if samples >= n_scenes:
                    break

                obs = obs_next

            if samples >= n_scenes:
                break

    def transform_angle(self):
        assert self.file_type == 'pkl', 'For now only pkl'
        dataset_path = os.path.join(self.log_dir, 'dataset.' + self.file_type)
        dataset_file = open(dataset_path, 'rb')
        data = pickle.load(dataset_file)

        for i in range(len(data[0])):
            theta = data[0][i][-1]
            if theta < 0:
                theta += 2 * np.pi
            theta -= np.pi / 4
            if theta < 0:
                theta += 2 * np.pi
            data[0][i][-1] = theta

        dataset_scaled_path = os.path.join(self.log_dir, 'dataset_transformed_angle.' + self.file_type)
        dataset_scaled_file = open(dataset_scaled_path, 'wb')
        pickle.dump(data, dataset_scaled_file)
        dataset_file.close()
        dataset_scaled_file.close()

    def update(self):
        assert self.file_type == 'pkl', 'For now only pkl'
        dataset_path = os.path.join(self.log_dir, 'dataset.' + self.file_type)
        dataset_file = open(dataset_path, 'rb')
        data = pickle.load(dataset_file)

        data[0] = np.delete(data[0], [256, 257, 258, 259], axis=1)

        dataset_scaled_path = os.path.join(self.log_dir, 'dataset_remove.' + self.file_type)
        dataset_scaled_file = open(dataset_scaled_path, 'wb')
        pickle.dump(data, dataset_scaled_file)
        dataset_file.close()
        dataset_scaled_file.close()

if __name__ == '__main__':
    pid = os.getpid()
    print('Process ID:', pid)
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
    params['env']['params']['vae_path'] = os.path.join(logging_dir, 'VAE')

    exp = PushObstacleSupervisedExp(params=params,
                                    log_dir=os.path.join(logging_dir, 'push_obstacle_supervised'),
                                    vae_path=os.path.join(logging_dir, 'VAE'),
                                    actor_path=os.path.join(logging_dir, 'push_obstacle_supervised/actor_deterministic_256size/model_60.pkl'),
                                    # actor_path=None,
                                    seed=4,
                                    file_type='pkl',
                                    partial_dataset=None)
    # exp.collect_samples(n_samples=8000, plot=True)
    # exp.create_dataset(rotations=1)
    # exp.merge_datasets(seeds=[0, 1])
    # exp.scale_outputs()
    # exp.train(hyperparams={'device': 'cpu',
    #                        'scaler': ['standard', None],
    #                        'learning_rate': 0.001,
    #                        'batch_size': 8,
    #                        'loss': 'cos',
    #                        'hidden_units': [400, 300],
    #                        'weight_decay': 0,
    #                        'dropout': 0.0},
    #           epochs=150,
    #           save_every=10,
    #           suffix='_deterministic_256size')
    exp.visualize_dataset()
    # exp.visual_evaluation(plot=True)
    # exp.eval_in_scenes(n_scenes=1000, random_policy=False)
    # exp.transform_angle()
    # exp.update()
