import unittest
import logging
import gym
from robamine.envs.clutter import Clutter
import pickle
import os
import numpy as np


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
        self.assertEqual(info['extra_data']['displacement'][0], 0.25)
        self.assertEqual(info['extra_data']['displacement'][1], 0.0)
        self.assertEqual(info['extra_data']['displacement'][2][0], 0.15021500375833308)
        self.assertEqual(info['extra_data']['displacement'][2][1], 0.04533153404584551)
        self.assertEqual(info['extra_data']['displacement'][2][2], 0.5600061048286464)

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

    def test_predict_displacement_from_forces(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_predict_displacement_from_forces.pkl'), 'rb') as file:
            pkl = pickle.load(file)
        env_params = pkl['env_params']
        env_params['params']['render'] = False
        # state_dict = pkl['state_dict']
        env = gym.make(env_params['name'], params=env_params['params'])
        # env.seed(0)

        # # Save after env.reset()
        # with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_predict_displacement_from_forces.pkl'), 'wb') as file:
        #     pickle.dump({'env_params': env_params, 'state_dict': env.state_dict()}, file)

        # Empty push should return zero displacement
        action = 8
        env.reset()
        env.step(action)
        np.testing.assert_array_equal(env.predicted_displacement_push_step, np.zeros(3))

        # Test with actual pushes

        errors = np.empty((1, 3))
        env.seed(0)
        for i in range(20):
            env.reset()
            action = 0
            env.step(action)
            error = np.abs(env.predicted_displacement_push_step - env.target_displacement_push_step)
            errors = np.concatenate((errors, error[None, :]))
        mean_error = np.mean(errors, axis=0)

        self.assertTrue(mean_error[0] < 0.01)  # < 1cm
        self.assertTrue(mean_error[1] < 0.02)  # < 1cm
        self.assertTrue(mean_error[2] < 0.3)   # < 0.3 rads = 17 degress

    def test_outputs(self):
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
        env.seed(0)
        state = env.reset()
        self.assertEqual(state.shape, (263,))
        action = 0
        next_state, reward, done, info = env.step(action)
        self.assertEqual(next_state.shape, (263,))

        # Test info dict
        self.assertTrue(isinstance(info, dict))
        self.assertTrue('experience_time' in info)
        self.assertTrue('success' in info)
        self.assertTrue('extra_data' in info)
        self.assertTrue(isinstance(info['extra_data'], dict))
        self.assertTrue('displacement' in info['extra_data'])
        self.assertTrue('push_forces_vel' in info['extra_data'])
        self.assertTrue(isinstance(info['extra_data']['displacement'], list))
        self.assertEqual(len(info['extra_data']['displacement']), 3)
        self.assertTrue(isinstance(info['extra_data']['push_forces_vel'], list))
        self.assertTrue(len(info['extra_data']['push_forces_vel']) == 2)
        self.assertTrue(isinstance(info['extra_data']['push_forces_vel'][0], np.ndarray))
        self.assertEqual(info['extra_data']['push_forces_vel'][0].shape, (1002, 3))
        self.assertTrue(isinstance(info['extra_data']['push_forces_vel'][1], np.ndarray))
        self.assertEqual(info['extra_data']['push_forces_vel'][1].shape, (1002, 3))

        env_params['params']['split'] = True
        env = gym.make(env_params['name'], params=env_params['params'])
        env.seed(0)
        state = env.reset()
        self.assertEqual(state.shape, (8 * 263,))
        action = 0
        next_state, reward, done, info = env.step(action)
        self.assertEqual(next_state.shape, (8 * 263,))

if __name__ == '__main__':
    unittest.main()
