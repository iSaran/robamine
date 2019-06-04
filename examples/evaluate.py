import tensorflow as tf
import logging
import robamine as rm


if __name__ == '__main__':
    rm.rb_logging.init('/tmp/robamine_logs/')
    logger = logging.getLogger('robamine')

    loaded_agent = rm.DDPG.load('/tmp/robamine_logs/robamine_logs_2019.06.03.13.09.30.384616/DDPG_Clutter-v0/model.pkl')

    world = rm.World(loaded_agent, 'Clutter-v0')
    world.evaluate(n_episodes=100, print_progress_every=1, render=True)
