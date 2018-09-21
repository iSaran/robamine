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
            inputs = [[3, 3, 3], [2, 2, 2]]
            network_output = sess.run(actor.out, feed_dict = {actor.inputs: inputs})
            true = np.reshape([1.3060998e-06, 3.5555320e-07, 8.7073329e-07, 2.3703552e-07], (2,2))
            self.assertTrue(np.allclose(network_output, true))

            grad_q = [[1, 1], [1, 1]]
            unnormalized_gradient = sess.run(actor.unnormalized_gradients, feed_dict = {actor.inputs: inputs, actor.grad_q_wrt_a: grad_q})
            self.assertTrue(isinstance(unnormalized_gradient, list))
            self.assertEqual(len(unnormalized_gradient), len(param_name))
            true = [[0., -0.00014419, 0.], [0., -0.00014419, 0.], [0., -0.00014419, 0.]]
            self.assertTrue(np.allclose(unnormalized_gradient[0], true))

    def test_target_actor(self):
        with tf.Session() as sess:
            # Se a random seed to have a reproducable test every time
            tf.set_random_seed(1)

            # Create an actor and its target network
            actor = Actor(sess, input_dim=3, hidden_dims=[3, 4], out_dim=2, final_layer_init=[-0.003, 0.003], batch_size=10, learning_rate=1e-3)
            tau = 0.01
            target_actor = TargetActor(actor, tau)
            sess.run(tf.global_variables_initializer())

            # Equalize the parameters of the two networks and test if this is happening
            target_actor.equalize_params()
            self.assertTrue(np.allclose(sess.run(target_actor.actor_net_params[0]), sess.run(target_actor.net_params[0])))

            # Change some parameters of the network, like some training happened and update the target
            actor.net_params[0].assign(np.reshape([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], (3,3))).eval()
            prev_value_target = sess.run(target_actor.net_params[0][0][0])
            next_value_target = sess.run(target_actor.actor_net_params[0][0][0])
            target_actor.update_params()
            expected_value = tau * next_value_target + (1 - tau) * prev_value_target
            true_value = sess.run(target_actor.net_params[0][0][0])
            self.assertAlmostEqual(expected_value, true_value)

if __name__ == '__main__':
    unittest.main()
