import argparse
import tensorflow as tf
import logging
import robamine as rm
import torch
from robamine.algo.ddpgtorch import default_params
import yaml


def run(file):
    with open("../yaml/lwr-ddpg.yml", 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            rm.rb_logging.init(directory=params['logging_directory'], file_level=logging.INFO)
            logger = logging.getLogger('robamine')
            world = rm.World.from_dict(params)
            if params['mode'] == 'Train & Evaluate':
                world.train_and_eval(n_episodes_to_train=params['train']['episodes'], \
                                     n_episodes_to_evaluate=params['train']['episodes'], \
                                     evaluate_every=params['evaluate_every'], \
                                     save_every=params['save_every'], \
                                     print_progress_every=10, \
                                     render_train=params['train']['render'], \
                                     render_eval=params['eval']['render'])
            elif params['mode'] == 'Train':
                world.train(n_episodes=params['train']['episodes'], \
                            print_progress_every=1, \
                            save_every=params['save_every'])
            elif params['mode'] == 'Evaluate':
                world.evaluate(n_episodes=params['eval']['episodes'], \
                               print_progress_every=1, \
                               save_every=params['save_every'])
            else:
                logger.error('The mode does not exist. Select btn Train, Train & Evaluate and Evaluate.')
        except yaml.YAMLError as exc:
            print(exc)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', type=str, default='lwr-ddpg', help='The id of the gym environment to use')
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

if __name__ == '__main__':
    args = parse_args()
    run(**args)
