import unittest

import tensorflow as tf
import numpy as np

from rlrl_py.algo.ddpg import DDPG, Actor, TargetActor, Critic, TargetCritic, ReplayBuffer

class TestAgent(unittest.TestCase):
    def test_construct(self):
        with tf.Session() as sess:
            agent = DDPG(sess, 'SphereReacher-v1').train(10, render=False)

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
            network_output = actor.predict(inputs)
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

    def test_critic(self):
        with tf.Session() as sess:
            tf.set_random_seed(1)
            # Create an actor
            critic = Critic(sess, input_dim=(3, 2), hidden_dims=(3, 4))
            sess.run(tf.global_variables_initializer())

            self.assertTrue(isinstance(critic.inputs, tuple))
            self.assertEqual(critic.inputs[0].shape[1], 3)
            self.assertEqual(critic.inputs[0].name, 'CriticInputData/X:0')
            self.assertEqual(critic.inputs[1].shape[1], 2)
            self.assertEqual(critic.inputs[1].name, 'CriticInputData_1/X:0')

            self.assertTrue(isinstance(critic.out, tf.Tensor))
            self.assertEqual(critic.out.shape[1], 1)
            self.assertEqual(critic.out.name, 'CriticFullyConnected_3/BiasAdd:0')

            # Test that the names and the shapes of the network's parameters are correct
            param_name = ['CriticFullyConnected/W:0', 'CriticFullyConnected/b:0',
                    'CriticBatchNormalization/beta:0',
                    'CriticBatchNormalization/gamma:0',
                    'CriticFullyConnected_1/W:0', 'CriticFullyConnected_1/b:0',
                    'CriticFullyConnected_2/W:0', 'CriticFullyConnected_2/b:0',
                    'CriticFullyConnected_3/W:0', 'CriticFullyConnected_3/b:0']
            param_shape = [(3, 3), (3,), (3,), (3,), (3, 4), (4,), (2, 4), (4,), (4, 1), (1,)]
            self.assertEqual(len(critic.net_params), len(param_name))

            for i in range(len(param_name)):
                self.assertEqual(critic.net_params[i].name, param_name[i])
                self.assertEqual(critic.net_params[i].shape, param_shape[i])

            # Test that the op works
            state_input = np.reshape([[3, 3, 3], [2, 2, 2]], (2, 3))
            action_input = np.reshape([[3, 3], [2, 2]], (2, 2))
            inputs = (state_input, action_input)
            network_output = critic.predict(inputs)
            true = np.reshape([-2.6877206e-06, -1.7918137e-06], (2,1))
            self.assertTrue(np.allclose(network_output, true))

            grad = critic.get_grad_q_wrt_actions(inputs)
            true = np.reshape([-3.5411253e-05,  3.6719161e-05, -3.5411253e-05,  3.6719161e-05], (2,2))
            self.assertTrue(np.allclose(grad, true))

    def test_target_critic(self):
        with tf.Session() as sess:
            # Se a random seed to have a reproducable test every time
            tf.set_random_seed(1)

            # Create an critic and its target network
            critic = Critic(sess, input_dim=(3, 2), hidden_dims=(3, 4))
            tau = 0.01
            target_critic = TargetCritic(critic, tau)
            sess.run(tf.global_variables_initializer())

            # Equalize the parameters of the two networks and test if this is happening
            target_critic.equalize_params()
            self.assertTrue(np.allclose(sess.run(target_critic.critic_net_params[0]), sess.run(target_critic.net_params[0])))

            # Change some parameters of the network, like some training happened and update the target
            critic.net_params[0].assign(np.reshape([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], (3,3))).eval()
            prev_value_target = sess.run(target_critic.net_params[0][0][0])
            next_value_target = sess.run(target_critic.critic_net_params[0][0][0])
            target_critic.update_params()
            expected_value = tau * next_value_target + (1 - tau) * prev_value_target
            true_value = sess.run(target_critic.net_params[0][0][0])
            self.assertAlmostEqual(expected_value, true_value)

class TestReplayBuffer(unittest.TestCase):
    def test_creating_object(self):
        replay_buffer = ReplayBuffer(10)

    def test_store(self):
        transitions = []
        transitions.append({'state': [1, 0, 0, 0], 'action': [1, 4], 'reward': -0.30, 'next_state': [1, 40, 3, 4], 'terminal': 0.0 })
        transitions.append({'state': [2, 2, 0, 0], 'action': [2, 3], 'reward': -10.3, 'next_state': [2, 40, 3, 4], 'terminal': 0.0 })
        transitions.append({'state': [3, 8, 1, 0], 'action': [3, 2], 'reward': -20.3, 'next_state': [3, 40, 3, 4], 'terminal': 1.0 })
        transitions.append({'state': [4, 3, 1, 1], 'action': [4, 1], 'reward': -30.3, 'next_state': [4, 40, 3, 4], 'terminal': 0.0 })

        replay_buffer = ReplayBuffer(10)

        for t in transitions:
            replay_buffer.store(t['state'], t['action'], t['reward'], t['next_state'], t['terminal'])

        self.assertEqual(replay_buffer.size(), 4)
        self.assertEqual(replay_buffer(2)[0], [3, 8, 1, 0])

    def test_sample(self):
        transitions = []
        transitions.append({'state': [1, 0, 0, 0], 'action': [1, 4], 'reward': -0.30, 'next_state': [1, 40, 3, 4], 'terminal': 0.0 })
        transitions.append({'state': [2, 2, 0, 0], 'action': [2, 3], 'reward': -10.3, 'next_state': [2, 40, 3, 4], 'terminal': 0.0 })
        transitions.append({'state': [3, 8, 1, 0], 'action': [3, 2], 'reward': -20.3, 'next_state': [3, 40, 3, 4], 'terminal': 1.0 })
        transitions.append({'state': [4, 3, 1, 1], 'action': [4, 1], 'reward': -30.3, 'next_state': [4, 40, 3, 4], 'terminal': 0.0 })

        replay_buffer = ReplayBuffer(10, 1)

        for t in transitions:
            replay_buffer.store(t['state'], t['action'], t['reward'], t['next_state'], t['terminal'])

        state_batch, action_batch, reward_batch, next_state_batch, terminal = replay_buffer.sample_batch(2)

        self.assertTrue((state_batch[0] == np.array([2, 2, 0, 0])).all())
        self.assertTrue((state_batch[1] == np.array([3, 8, 1, 0])).all())
        self.assertTrue((action_batch[0] == np.array([2, 3])).all())
        self.assertTrue((action_batch[1] == np.array([3, 2])).all())
        self.assertEqual(reward_batch[0], -10.3)
        self.assertEqual(reward_batch[1], -20.3)
        self.assertTrue((next_state_batch[0] == np.array([2, 40, 3, 4])).all())
        self.assertTrue((next_state_batch[1] == np.array([3, 40, 3, 4])).all())
        self.assertEqual(terminal[0], 0.0)
        self.assertEqual(terminal[1], 1.0)

if __name__ == '__main__':
    unittest.main()
