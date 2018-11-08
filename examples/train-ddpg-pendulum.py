import tensorflow as tf
import logging

from robamine.algo.core import World, WorldMode
from robamine.algo.ddpg import DDPG, DDPGParams
from robamine import rb_logging
from robamine.algo.util import seed_everything

if __name__ == '__main__':

    rb_logging.init('/home/iason/robamine_logs/ddpg-pendulum', file_level=logging.INFO)
    logger= logging.getLogger('robamine')
    seed_everything(999)

    world = World.create(DDPGParams(suffix='x'), 'Pendulum-v0')
    world.evaluate(n_episodes=1000, render=True)
    # world.agent.save('/home/iason/hahaha.pkl')
#
    ###########################world2 = World.create(DDPGParams(suffix="_2"), 'Pendulum-v0')
    ###########################logger.info('Parameters of second world before training')
    ###########################logger.info(world2.agent.sess.run(world2.agent.actor.trainable_params))
    ###########################world2.agent.load('/home/iason/hahahah.pkl')
    ###########################logger.info('Parameters of after world before training')
    ###########################logger.info(world.agent.sess.run(world.agent.actor.trainable_params))
    #world.train(n_episodes=2)
#
    # with tf.Session() as sess:
    #     logger.info('Parameters of second world before training')

