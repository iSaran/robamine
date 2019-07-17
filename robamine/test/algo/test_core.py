import unittest

import tensorflow as tf
import numpy as np
import os

import logging
import robamine as rm
import gym

class TestWorld(unittest.TestCase):
    def test_new_world_from_strings(self):
        rm.rb_logging.init(console_level=logging.ERROR, file_level=logging.ERROR)  # Do not show info messages in unittests
        rm.seed_everything(999)

        with tf.variable_scope('test_new_world_from_strings'):
            world = rm.World('DDPG', 'Pendulum-v0')

            self.assertTrue(isinstance(world.agent, rm.DDPG))
            self.assertTrue(isinstance(world.env, gym.Env))
            self.assertEqual(world.env.spec.id, 'Pendulum-v0')
            self.assertEqual(world.agent.name, 'DDPG')
            self.assertEqual(world.agent.state_dim, 3)
            self.assertEqual(world.agent.action_dim, 1)

    def test_new_world_from_agent_params(self):
        rm.rb_logging.init(console_level=logging.ERROR, file_level=logging.ERROR)  # Do not show info messages in unittests
        rm.seed_everything(999)

        with tf.variable_scope('test_new_world_from_agent_params'):
            agent = rm.algo.ddpg.default_params
            agent['discount'] = 0.54
            world = rm.World(agent, 'Pendulum-v0')

            self.assertTrue(isinstance(world.agent, rm.DDPG))
            self.assertTrue(isinstance(world.env, gym.Env))
            self.assertEqual(world.env.spec.id, 'Pendulum-v0')
            self.assertEqual(world.agent.name, 'DDPG')
            self.assertEqual(world.agent.state_dim, 3)
            self.assertEqual(world.agent.action_dim, 1)
            self.assertEqual(world.agent.params['discount'], 0.54)

    def test_new_world_from_agent_object(self):
        rm.rb_logging.init(console_level=logging.ERROR, file_level=logging.ERROR)  # Do not show info messages in unittests
        rm.seed_everything(999)

        with tf.variable_scope('test_new_world_from_agent_object'):
            ddpg = rm.DDPG(state_dim=3, action_dim=1)
            world = rm.World(ddpg, 'Pendulum-v0')
            self.assertTrue(isinstance(world.agent, rm.DDPG))
            self.assertTrue(isinstance(world.env, gym.Env))
            self.assertEqual(world.env.spec.id, 'Pendulum-v0')
            self.assertEqual(world.agent.name, 'DDPG')
            self.assertEqual(world.agent.state_dim, 3)
            self.assertEqual(world.agent.action_dim, 1)

    def test_new_world_failures(self):
        rm.rb_logging.init(console_level=logging.ERROR, file_level=logging.ERROR)  # Do not show info messages in unittests
        rm.seed_everything(999)

        with tf.variable_scope('test_new_world_failures'):
            placeholder = 0
            # rm.World(rm.DDPG(state_dim=2, action_dim=1), 'Pendulum-v0')
            self.assertRaises(AssertionError, rm.World, rm.DDPG(2, 1), 'Pendulum-v0')

if __name__ == '__main__':
    unittest.main()
