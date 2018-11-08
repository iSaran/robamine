import unittest

import tensorflow as tf
import numpy as np
import os

from robamine.algo.ddpg import DDPG, DDPGParams, Actor, ActorParams, Target, Critic, CriticParams, ReplayBuffer
from robamine.algo.util import Plotter, seed_everything
from robamine import rb_logging
import logging

class TestAgent(unittest.TestCase):
    def test_reproducability_with_pendulum(self):
        with tf.Session() as sess:
            rb_logging.init(console_level=logging.WARN)  # Do not show info messages in unittests
            seed_everything(999)
            agent = DDPG(sess, 'Pendulum-v0', random_seed=999, actor_gate_gradients = True)
            agent.train(n_episodes=20, episode_batch_size=5, episodes_to_evaluate=5)

            streams = ['train_episode', 'train_batch', 'eval_episode', 'eval_batch']
            pl = Plotter(agent.logger.log_path, streams)
            pl_2 = Plotter(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robamine_logs_2018.11.05.12.29.32.260511/DDPG_Pendulum-v0'), streams)

            for stream in streams:
                x, y = pl.extract_data(stream)
                x_d, y_d = pl_2.extract_data(stream)
                for y_var in y_d:
                    error = np.array(y_d[y_var]) - np.array(y[y_var])
                    self.assertEqual(np.max(error), 0.0, 'max error not zero during processing {0} in stream {1}'.format(y_var, stream))

class TestActorCritic(unittest.TestCase):
    def test_actor(self):
        with tf.Session() as sess:
            tf.set_random_seed(1)
            # Create an actor
            actor = Actor.create(sess, ActorParams(state_dim=3, hidden_units=(3, 4), action_dim=2, batch_size=10, learning_rate=1e-4))
            sess.run(tf.global_variables_initializer())

            # Test that the names and the shapes of the network's parameters are correct
            param_name = ['Actor/network/dense/kernel:0', 'Actor/network/dense/bias:0',
                    'Actor/network/batch_normalization/gamma:0',
                    'Actor/network/batch_normalization/beta:0',
                    'Actor/network/dense_1/kernel:0', 'Actor/network/dense_1/bias:0',
                    'Actor/network/batch_normalization_1/gamma:0',
                    'Actor/network/batch_normalization_1/beta:0',
                    'Actor/network/dense_2/kernel:0', 'Actor/network/dense_2/bias:0']
            param_shape = [(3, 3), (3,), (3,), (3,), (3, 4),
                    (4,), (4,), (4,), (4, 2), (2,)]
            self.assertEqual(len(actor.net_params), len(param_name))

            for i in range(len(param_name)):
                self.assertEqual(actor.net_params[i].name, param_name[i])
                self.assertEqual(actor.net_params[i].shape, param_shape[i])

            # Test that the op works
            inputs = [[3, 3, 3], [2, 2, 2]]
            network_output = actor.predict(inputs)
            true = np.reshape([5.0694158e-04, -2.0009999e-03, 2.1590035e-05, -9.6079556e-04], (2,2))
            self.assertTrue(np.allclose(network_output, true))

    def test_target_actor(self):
        with tf.Session() as sess:
            # Se a random seed to have a reproducable test every time
            tf.set_random_seed(1)

            # Create an actor and its target network
            actor = Actor.create(sess, 3, hidden_dims=[3, 4], out_dim=2, final_layer_init=[-0.003, 0.003], batch_size=10, learning_rate=1e-3, name='2')
            tau = 0.01
            target_actor = Target.create(actor, tau)
            sess.run(tf.global_variables_initializer())

            # Equalize the parameters of the two networks and test if this is happening
            target_actor.equalize_params()
            self.assertTrue(np.allclose(sess.run(target_actor.base_net_params[0]), sess.run(target_actor.net_params[0])))

            # Change some parameters of the network, like some training happened and update the target
            actor.net_params[0].assign(np.reshape([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], (3,3))).eval()
            prev_value_target = sess.run(target_actor.net_params[0][0][0])
            next_value_target = sess.run(target_actor.base_net_params[0][0][0])
            target_actor.update_params()
            expected_value = tau * next_value_target + (1 - tau) * prev_value_target
            true_value = sess.run(target_actor.net_params[0][0][0])
            self.assertAlmostEqual(expected_value, true_value, places=6)

    def test_critic(self):
        with tf.Session() as sess:
            tf.set_random_seed(1)
            # Create an actor
            critic = Critic.create(sess, 3, 2, hidden_dims=(3, 4), name='1')
            sess.run(tf.global_variables_initializer())

            self.assertEqual(critic.state_input.shape[1], 3)
            self.assertEqual(critic.state_input.name, 'ddpg_critic_1/state_input:0')
            self.assertEqual(critic.action_input.shape[1], 2)
            self.assertEqual(critic.action_input.name, 'ddpg_critic_1/action_input:0')

            self.assertTrue(isinstance(critic.out, tf.Tensor))
            self.assertEqual(critic.out.shape[1], 1)
            self.assertEqual(critic.out.name, 'ddpg_critic_1/network/dense_2/BiasAdd:0')

            # Test that the names and the shapes of the network's parameters are correct
            param_name = ['ddpg_critic_1/network/dense/kernel:0', 'ddpg_critic_1/network/dense/bias:0',
                    'ddpg_critic_1/network/batch_normalization/gamma:0',
                    'ddpg_critic_1/network/batch_normalization/beta:0',
                    'ddpg_critic_1/network/dense_1/kernel:0', 'ddpg_critic_1/network/dense_1/bias:0',
                    'ddpg_critic_1/network/batch_normalization_1/gamma:0',
                    'ddpg_critic_1/network/batch_normalization_1/beta:0',
                    'ddpg_critic_1/network/dense_2/kernel:0', 'ddpg_critic_1/network/dense_2/bias:0']
            param_shape = [(3, 3), (3,), (3,), (3,), (5, 4), (4,), (4), (4,), (4, 1), (1,)]
            self.assertEqual(len(critic.net_params), len(param_name))

            for i in range(len(param_name)):
                self.assertEqual(critic.net_params[i].name, param_name[i], 'while processing ' + param_name[i])
                self.assertEqual(critic.net_params[i].shape, param_shape[i], 'while processing ' + param_name[i])

            # Test that the op works
            state_input = np.reshape([[3, 3, 3], [2, 2, 2]], (2, 3))
            action_input = np.reshape([[3, 3], [2, 2]], (2, 2))
            network_output = critic.predict(state_input, action_input)
            true = np.reshape([-0.00185596, -0.00069298], (2,1))
            self.assertTrue(np.allclose(network_output, true))

            grad = critic.get_grad_q_wrt_actions(state_input, action_input)
            true = np.reshape([-0.0005927,  -0.00057028, -0.0005927,  -0.00057028], (2,2))
            self.assertTrue(np.allclose(grad, true))

    def test_target_critic(self):
        with tf.Session() as sess:
            # Se a random seed to have a reproducable test every time
            tf.set_random_seed(1)

            # Create an critic and its target network
            critic = Critic.create(sess, 3, 2, hidden_dims=(3, 4), name='2')
            tau = 0.01
            target_critic = Target.create(critic, tau)
            sess.run(tf.global_variables_initializer())

            # Equalize the parameters of the two networks and test if this is happening
            target_critic.equalize_params()
            self.assertTrue(np.allclose(sess.run(target_critic.base_net_params[0]), sess.run(target_critic.net_params[0])))

            # Change some parameters of the network, like some training happened and update the target
            critic.net_params[0].assign(np.reshape([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], (3,3))).eval()
            prev_value_target = sess.run(target_critic.net_params[0][0][0])
            next_value_target = sess.run(target_critic.base_net_params[0][0][0])
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
        replay_buffer.seed(999)

        for t in transitions:
            replay_buffer.store(t['state'], t['action'], t['reward'], t['next_state'], t['terminal'])

        state_batch, action_batch, reward_batch, next_state_batch, terminal = replay_buffer.sample_batch(2)

        self.assertTrue((state_batch[0] == np.array([1, 0, 0, 0])).all())
        self.assertTrue((state_batch[1] == np.array([3, 8, 1, 0])).all())
        self.assertTrue((action_batch[0] == np.array([1, 4])).all())
        self.assertTrue((action_batch[1] == np.array([3, 2])).all())
        self.assertEqual(reward_batch[0], -0.3)
        self.assertEqual(reward_batch[1], -20.3)
        self.assertTrue((next_state_batch[0] == np.array([1, 40, 3, 4])).all())
        self.assertTrue((next_state_batch[1] == np.array([3, 40, 3, 4])).all())
        self.assertEqual(terminal[0], 0.0)
        self.assertEqual(terminal[1], 1.0)

if __name__ == '__main__':
    unittest.main()
