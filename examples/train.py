import tensorflow as tf
import logging
import robamine as rm


if __name__ == '__main__':
    rm.rb_logging.init('/tmp/robamine_logs/')
    logger = logging.getLogger('robamine')
    world = rm.World(rm.DDPGParams(), 'Clutter-v1')
    world.train(n_episodes=1000, print_progress_every=1)
