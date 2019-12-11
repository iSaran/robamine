import unittest
from robamine.algo.splitdynamicsmodelpose import SplitDynamicsModelPose
from robamine.algo.util import EnvData, Transition
import os
import numpy.testing as np_test
import numpy as np
import torch

# Env staff for integration
import gym

class TestAgent(unittest.TestCase):
    def test_init(self):
        params = {
            'action_dim': 16,
            'state_dim': 2120,
            'batch_size': [64, 64],
            'device': 'cpu',
            'hidden_units': [[20], [20]],
            'learning_rate': [0.001, 0.001],
            'loss': ['mse', 'mse'],
            'n_epochs': [1000, 1000]
        }
        model = SplitDynamicsModelPose(params)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model.pkl')
        model.save(path)

        model_loaded = SplitDynamicsModelPose.load(path)

        self.assertEqual(model.inputs, model_loaded.inputs)
        self.assertEqual(model.outputs, model_loaded.outputs)

    def test_learn(self):
        # Tested in the integrated test with Clutter Env
        pass

    def test_predict(self):
        # Tested in the integrated test with Clutter Env
        pass

class TestIntegrationWithClutterEnv(unittest.TestCase):
    def test_learn(self):

        # Create env and compile EnvData for pushes without obstacles
        # -----------------------------------------------------------

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
          'split': True,
          'extra_primitive': False,
          'all_equal_height_prob': 0.0,
          'finger_size': [0.005, 0.005]
          }
        }
        env = gym.make(env_params['name'], params=env_params['params'])
        env.seed(0)
        env_data = EnvData(['extra_data'])
        action = [0, 1, 2, 8, 8]
        for i in range(5):
            state = env.reset()
            next_state, reward, done, info = env.step(action[i])
            transition = Transition(state, action[i], reward, next_state, done)
            env_data.transitions.append(transition)
            if 'extra_data' in info:
                env_data.info['extra_data'].append(info['extra_data'])


        # Create model, load dataset from clutter and learn for one epoch
        # ---------------------------------------------------------------

        params = {
            'action_dim': 16,
            'state_dim': 2120,
            'batch_size': [64, 64],
            'device': 'cpu',
            'hidden_units': [[20], [20]],
            'learning_rate': [0.001, 0.001],
            'loss': ['mse', 'mse'],
            'n_epochs': [1000, 1000]
        }
        torch.manual_seed(0)
        model = SplitDynamicsModelPose(params)
        model.load_dataset(env_data)
        model.learn()

        # Test predictions of the learned model
        # -------------------------------------
        vel = env_data.info['extra_data'][1]['push_finger_vel']
        force = env_data.info['extra_data'][1]['push_finger_forces']
        action = 1
        prediction = model.predict([vel, force], action)

        expected = np.array([0.04078154265880585, 0.03126578405499458, 0.11294207721948624])
        np_test.assert_equal(prediction, expected)


if __name__ == '__main__':
    unittest.main()
# Datapoint[x= [0.06328722 1.57079633], y= [ 0.00518818  0.05335347 -0.03774902]]
