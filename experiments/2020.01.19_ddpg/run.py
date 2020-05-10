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

    process_episodes(os.path.join(params['world']['logging_dir'], exp_dir))
