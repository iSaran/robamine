import argparse
import tensorflow as tf
import logging
import robamine as rm
import torch


def run(env_id, agent_id, episodes):
    rm.rb_logging.init(directory='/tmp/robamine_logs/', file_level=logging.INFO)
    logger = logging.getLogger('robamine')
    torch.manual_seed(0)
    world = rm.World(agent_id, env_id)
    world.seed(0)
    world.train(n_episodes=episodes, print_progress_every=100, save_every=100, render=False)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env-id', type=str, default='Clutter-v0', help='The id of the gym environment to use')
    parser.add_argument('--agent-id', type=str, default='DDPGTorch', help='The id of the gym environment to use')
    parser.add_argument('--episodes', type=int, default=5000, help='The number of episodes to train')
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

if __name__ == '__main__':
    args = parse_args()
    run(**args)
