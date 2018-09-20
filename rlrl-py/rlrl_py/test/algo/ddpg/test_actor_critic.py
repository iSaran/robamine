import unittest
from rlrl_py.algo.ddpg.actor_critic import Actor, TargetActor, Critic, TargetCritic
import tensorflow as tf
import numpy as np

class TestActorCritic(unittest.TestCase):
    def test_actor(self):
        with tf.Session() as sess:
            tf.set_random_seed(1)
            # Create an actor
            actor = Actor(sess, input_dim=3, hidden_dims=[3, 4], out_dim=2, final_layer_init=[-0.003, 0.003], batch_size=10, learning_rate=1e-3)
            sess.run(tf.global_variables_initializer())

            # Test that the names and the shapes of the network's parameters are correct
            param_name = ['ActorFullyConnected/W:0', 'ActorFullyConnected/b:0',
                    'ActorBatchNormalization/beta:0',
                    'ActorBatchNormalization/gamma:0',
                    'ActorFullyConnected_1/W:0', 'ActorFullyConnected_1/b:0',
                    'ActorBatchNormalization_1/beta:0',
                    'ActorBatchNormalization_1/gamma:0',
                    'ActorFullyConnected_2/W:0', 'ActorFullyConnected_2/b:0']
            param_shape = [(3, 3), (3,), (3,), (3,), (3, 4),
                    (4,), (4,), (4,), (4, 2), (2,)]
            self.assertEqual(len(actor.net_params), len(param_name))

            for i in range(len(param_name)):
                self.assertEqual(actor.net_params[i].name, param_name[i])
                self.assertEqual(actor.net_params[i].shape, param_shape[i])

            # Test that the op works
            #print(sess.run(actor.grad_q_wrt_a, feed_dict = {actor.grad_q_wrt_a: [[3, 3]]}))
            #print(sess.run(actor.out, feed_dict = {actor.inputs: [[3, 3, 3], [2, 2, 2]]}))
            k = sess.run(actor.unnormalized_gradients, feed_dict = {actor.inputs: [[2, 2, 2]], actor.grad_q_wrt_a: [[3, 3]]})
            print('shape:', k.shape)
            for i in range(10):
                print('----')
                print(k[i])




if __name__ == '__main__':
    unittest.main()
