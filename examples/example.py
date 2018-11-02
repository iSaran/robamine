import tensorflow as tf
from rlrl_py.algo.ddpg import DDPG, Actor, Critic, Target
from rlrl_py.algo.util import seed_everything, Logger
import numpy as np
import logging


if __name__ == '__main__':
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:

        # log = logging.getLogger('tensorflow')
        # log.setLevel(logging.DEBUG)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # fh = logging.FileHandler('tensorflow.log')
        # fh.setLevel(logging.DEBUG)
        # fh.setFormatter(formatter)
        # log.addHandler(fh)

        seed_everything(999)

        logger = Logger(sess, '/home/iason/rlrl_logs/example', 'none', 'lol', True)
        # actor = Actor.create(sess, 2, [100, 100], 3)

        inp = tf.placeholder(tf.float64, [None, 10])
        out, net_params = Actor.architecture(inp, [100, 200], 10, [-3e-6, 3e-6])
        grad = tf.gradients(out, net_params)
        gradients = list(map(lambda x: tf.div(x, 10, name='div_by_N'), grad))
        optimizer = tf.train.AdamOptimizer(1e-4, name='optimizer').apply_gradients(zip(gradients, net_params))
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            # logger.console.debug('initial:' + actor.get_params().__str__())

            state_batch = np.random.rand(10, 10)
            outt = np.random.rand(10, 5)
            logger.console.debug('state_batch:' + state_batch.__str__())

            sess.run(optimizer, feed_dict={inp: state_batch})
            params= sess.run(net_params)
            logger.console.debug('result:' + params.__str__())


            ######## grad = np.random.rand(10, 3)
            ######## logger.console.debug('grad:' + grad.__str__())
            ######## # actor.learn(state_batch, grad)
            ######## logger.console.debug('final:' + actor.get_params().__str__())

