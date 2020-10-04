from robamine.algo.core import EvalWorld, TrainEvalWorld
from robamine.algo.splitddpg import SplitDDPG, Critic, Actor, ObstacleAvoidanceLoss
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

import robamine.envs.clutter_utils as clutter
import h5py
from robamine.utils.memory import ReplayBuffer
from robamine.algo.util import OrnsteinUhlenbeckActionNoise, NormalNoise, Transition

logger = logging.getLogger('robamine')


# General functions for training and rendered eval
# ------------------------------------------------

class SplitDDPGNoObsAvoid(SplitDDPG):
    '''
    In info dict it saves for each primitive the loss of the critic and the loss
    of the actor and the qvalues for each primitive. The q values are updated
    only in predict, so during training if you call explore the q values will be
    invalid.
    '''

    def __init__(self, state_dim, action_dim, params):
        self.hardcoded_primitive = params['env_params']['hardcoded_primitive']
        self.real_state = params.get('real_state', False)
        self.state_dim = clutter.get_observation_dim(self.hardcoded_primitive, real_state=self.real_state)
        self.action_dim = [3]
        super(SplitDDPG, self).__init__(self.state_dim, self.action_dim, 'SplitDDPG', params)
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
            self.actor_state_dim = clutter.RealState.dim()

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

        self.max_init_distance = self.params['env_params']['push']['target_init_distance'][1]
        if self.params.get('save_heightmaps_disk', False):
            self.file_heightmaps = h5py.File(os.path.join(self.log_dir, 'heightmaps_dataset.h5py'), "a")
            self.file_heightmaps.create_dataset('heightmap_mask', (5000, 2, 386, 386), dtype='f')
            self.file_heightmaps_counter = 0

def train_eval(params):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='push_target_without_obs_avoid', file_level=logging.INFO)
    agent = SplitDDPGNoObsAvoid(None, None, params['agent']['params'])
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

def eval_in_scenes(params, dir, n_scenes=1000):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='', file_level=logging.INFO)
    with open(os.path.join(dir, 'config.yml'), 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config['env']['params']['render'] = False
    config['env']['params']['safe'] = True
    config['env']['params']['log_dir'] = params['world']['logging_dir']
    config['env']['params']['deterministic_policy'] = True
    config['env']['params']['nr_of_obstacles'] = [1, 13]
    config['world']['episodes'] = n_scenes
    world = EvalWorld.load(dir, overwrite_config=config)
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
    params['agent']['params']['actor']['autoencoder']['model'] = os.path.join(os.path.join(logging_dir, 'VAE'), 'model.pkl')
    params['agent']['params']['actor']['autoencoder']['scaler'] = os.path.join(os.path.join(logging_dir, 'VAE'), 'normalizer.pkl')
    params['env']['params']['push']['obstacle_avoid'] = False

    train_eval(params)


