import tensorflow as tf
import logging

from robamine.algo.core import World, WorldMode
from robamine.algo.ddpg import DDPG, DDPGParams
from robamine import rb_logging

if __name__ == '__main__':

    rb_logging.init('/home/iason/robamine_logs/ddpg-pendulum')

    with tf.Session() as sess:
        world = World.create(sess, DDPGParams(), 'Pendulum-v0')
        # world.train(n_episodes=1000, episode_batch_size=25, render=False, episodes_to_evaluate=10, render_eval=False)
        world.train(n_episodes=1000)
