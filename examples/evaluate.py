import tensorflow as tf
import logging
import robamine as rm


if __name__ == '__main__':
    rm.rb_logging.init('/tmp/robamine_logs/')
    logger = logging.getLogger('robamine')

    loaded_agent = rm.DDPG.load('/tmp/robamine_logs/robamine_logs_2019.05.29.11.44.44.175752/DDPG_ClutterPushOffTable-v0/model.pkl')



    world = rm.World(loaded_agent, 'ClutterPushOffTable-v0')
    world.evaluate(n_episodes=100, print_progress_every=1, render=True)
