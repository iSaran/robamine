import tensorflow as tf
import logging
from robamine.algo.core import World, WorldMode
from robamine.algo.dummy import Dummy, DummyParams
from robamine import rb_logging

if __name__ == '__main__':

    rb_logging.init('/home/iason/robamine_logs/ddpg-dummy')

    with tf.Session() as sess:
        world = World.create(sess, DummyParams(), 'Pendulum-v0')
        world.train(n_episodes=20, render=False)
        # agent = rb.algo.dummy.Dummy(sess)
        # agent.train(n_episodes=1000, episode_batch_size=25, render=False, episodes_to_evaluate=10, render_eval=False)
