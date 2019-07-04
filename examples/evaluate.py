import tensorflow as tf
import logging
import robamine as rm


if __name__ == '__main__':
    rm.rb_logging.init('/tmp/robamine_logs/')
    logger = logging.getLogger('robamine')
    world = rm.World.load('/tmp/robamine_logs/robamine_logs_2019.07.04.18.26.36.608486/DQN_CartPole-v0')
    world.evaluate(n_episodes=100, print_progress_every=1, render=True)
