import tensorflow as tf
import logging

from robamine.algo.core import World, WorldMode
from robamine.algo.dummy import Dummy, DummyParams
from robamine import rb_logging
import robamine as rm

if __name__ == '__main__':

    rm.rb_logging.init('/home/iason/robamine_logs/dummy-pendulum')
    # rb_logging.init('/home/iason/robamine_logs/dummy-pendulum', file_level=logging.INFO)

    world = rm.World('Dummy', 'Pendulum-v0')
    # world.train(n_episodes=20, render=False)
    # world.evaluate(n_episodes=20, render=False)
    world.train_and_eval(n_episodes_to_train=20, n_episodes_to_evaluate=10, evaluate_every=5)
    world.plot()
    # agent = rb.algo.dummy.Dummy(sess)
    # agent.train(n_episodes=1000, episode_batch_size=25, render=False, episodes_to_evaluate=10, render_eval=False)
