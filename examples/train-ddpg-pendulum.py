import tensorflow as tf
import logging

import robamine as rm

if __name__ == '__main__':

    rm.rb_logging.init('/home/iason/robamine_logs/ddpg-pendulum', file_level=logging.DEBUG)
    logger = logging.getLogger('robamine')
    rm.seed_everything(999)

    # world = rm.World(rm.DDPG.load('/home/iason/robamine_logs/ddpg-pendulum/robamine_logs_2018.11.09.15.24.03.602470/DDPG_SphereReacherShapedReward-v1/model.pkl'), 'SphereReacherShapedReward-v1')
    # world.evaluate(n_episodes=1000, render=True)

    world = rm.World(rm.DDPGParams(actor=rm.ActorParams(gate_gradients=True)), 'Pendulum-v0')
    world.train_and_eval(n_episodes_to_train=1000, n_episodes_to_evaluate=10, evaluate_every=25, save_every=10, print_progress_every=10, render_train=False)

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
