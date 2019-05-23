import tensorflow as tf
import logging
import robamine as rm


if __name__ == '__main__':
    rm.rb_logging.init('/tmp/robamine_logs/')
    logger = logging.getLogger('robamine')
    world = rm.World(rm.DDPGParams(exploration_noise='Normal'), 'Clutter-v0')
    world.train(n_episodes=500, print_progress_every=1, save_every=10)
