import unittest
import logging
import gym
from robamine.envs.clutter import Clutter
import pickle
import os


class TestClutter(unittest.TestCase):
    def test_get_target_displacement(self):
        env_params = {
        'name': 'Clutter-v0',
        'params': {
          'discrete': True,
          'nr_of_actions': 16,  # u x w
          'render': False,
          'nr_of_obstacles': [0, 0],
          'target_probability_box': 1.0,
          'target_height_range': [0.01, 0.01],
          'obstacle_probability_box': 1.0,
          'obstacle_height_range': [0.005, 0.005],
          'push_distance': [0.25, 0.25],
          'split': False,
          'extra_primitive': False,
          'all_equal_height_prob': 0.0,
          'finger_size': [0.005, 0.005]
          }
        }
        env = gym.make(env_params['name'], params=env_params['params'])
        state_dict = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_state_dict.pkl'), 'rb'))
        env.load_state_dict(state_dict)
        state = env.reset()
        #pickle.dump(env.state_dict(), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_state_dict.pkl'), 'wb'))
        action = 0
        next_state, reward, done, info = env.step(action)
        self.assertEqual(info['extra_data'][0], 0.25)
        self.assertEqual(info['extra_data'][1], 0.0)
        self.assertEqual(info['extra_data'][2][0], 0.15021500438099966)
        self.assertEqual(info['extra_data'][2][1], 0.04533153281382788)
        self.assertEqual(info['extra_data'][2][2], 0.5600061048269042)

    def test_push_distance_finger_size(self):
        # With push distance and finger deterministic we shouldn't have them in
        # the state
        env_params = {
        'name': 'Clutter-v0',
        'params': {
          'discrete': True,
          'nr_of_actions': 16,  # u x w
          'render': False,
          'nr_of_obstacles': [0, 0],
          'target_probability_box': 1.0,
          'target_height_range': [0.01, 0.01],
          'obstacle_probability_box': 1.0,
          'obstacle_height_range': [0.005, 0.005],
          'push_distance': [0.15, 0.15],
          'split': False,
          'extra_primitive': False,
          'all_equal_height_prob': 0.0,
          'finger_size': [0.005, 0.005]
          }
        }
        env = gym.make(env_params['name'], params=env_params['params'])
        state = env.reset()
        self.assertEqual(state.shape[0], 263)
        self.assertEqual(state.shape[0], env.observation_space.shape[0])

        # With push distance and finger deterministic we shouldn't have them in
        # the state
        env_params = {
        'name': 'Clutter-v0',
        'params': {
          'discrete': True,
          'nr_of_actions': 16,  # u x w
          'render': False,
          'nr_of_obstacles': [0, 0],
          'target_probability_box': 1.0,
          'target_height_range': [0.01, 0.01],
          'obstacle_probability_box': 1.0,
          'obstacle_height_range': [0.005, 0.005],
          'push_distance': [0.13, 0.15],
          'split': False,
          'extra_primitive': False,
          'all_equal_height_prob': 0.0,
          'finger_size': [0.003, 0.005]
          }
        }
        env = gym.make(env_params['name'], params=env_params['params'])
        state = env.reset()
        self.assertEqual(state.shape[0], 265)
        self.assertEqual(state.shape[0], env.observation_space.shape[0])


if __name__ == '__main__':
    unittest.main()
