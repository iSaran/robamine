from robamine.algo.core import TrainWorld, EvalWorld
# from robamine.algo.ddpg_torch import DDPG_TORCH
from robamine.algo.splitddpg import SplitDDPG
from robamine.algo.util import EpisodeListData
from robamine import rb_logging
import logging
import yaml
import sys
import socket
import numpy as np
import os
import gym

from robamine.algo.util import get_agent_handle
from robamine.envs.clutter_utils import RealState, plot_point_cloud_of_scene

logger = logging.getLogger('robamine')

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
        print('Seed:', seed)
        rng = np.random.RandomState()
        rng.seed(seed)
        obs = env.reset(seed=seed)
        # plot_point_cloud_of_scene(obs)
        while True:
            action = rng.uniform(-1, 1, 4)
            action[0] = 0
            obs, reward, done, info = env.step(action)
            # plot_point_cloud_of_scene(obs)
            print('reward: ', reward, 'done:', done)
            if done:
                break

def test():
    from robamine.envs.clutter_utils import discretize_2d_box, discretize_3d_box
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    points = discretize_3d_box(1, 0.5, 0.2, 0.1)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o')
    ax.axis('equal')
    plt.show()

if __name__ == '__main__':
    hostname = socket.gethostname()
    exp_dir = 'robamine_logs_dream_2020.05.08.16.36.58.36044'

    if hostname == 'dream':
        yml_name = 'params_dream.yml'
        logging_dir = '/home/espa/robamine_logs/'
    elif hostname == 'triss':
        yml_name = 'params_iason.yml'
        logging_dir = '/home/iason/robamine_logs/2020.01.16.split_ddpg'
    else:
        raise ValueError()
    with open(yml_name, 'r') as stream:
        params = yaml.safe_load(stream)

    params['world']['logging_dir'] = logging_dir

    # Run sth

    # train(params)
    #
    # eval_with_render(os.path.join(params['world']['logging_dir'], exp_dir))

    # process_episodes(os.path.join(params['world']['logging_dir'], exp_dir))
    check_transition(params)
    # test()
