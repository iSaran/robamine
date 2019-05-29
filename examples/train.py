import tensorflow as tf
import logging
import robamine as rm


if __name__ == '__main__':
    rm.rb_logging.init(directory='/tmp/robamine_logs/', file_level=logging.INFO)
    logger = logging.getLogger('robamine')
    world = rm.World(rm.DDPGParams(exploration_noise='Normal'), 'Clutter-v0')
    world.train(n_episodes=5000, print_progress_every=10, save_every=10)
