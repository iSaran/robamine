import argparse
import tensorflow as tf
import logging
import robamine as rm


def run(env_id, episodes):
    rm.rb_logging.init(directory='/tmp/robamine_logs/', file_level=logging.INFO)
    logger = logging.getLogger('robamine')
    world = rm.World(rm.DDPGParams(exploration_noise='Normal'), env_id)
    # world.train(n_episodes=episodes, print_progress_every=1, save_every=10)
    world.train_and_eval(n_episodes_to_train=episodes, n_episodes_to_evaluate=10, evaluate_every=50, render_train=True, render_eval=False, print_progress_every=1, save_every=10)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env-id', type=str, default='Clutter-v0', help='The id of the gym environment to use')
    parser.add_argument('--episodes', type=int, default=5000, help='The number of episodes to train')
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

if __name__ == '__main__':
    args = parse_args()
    run(**args)
