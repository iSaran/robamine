import unittest
from robamine.algo.splitdynamicsmodelposelstm import SplitDynamicsModelPoseLSTM
from robamine.algo.util import EnvData, Transition
import os
import numpy.testing as np_test
import numpy as np
import gym
import torch
import pickle

class TestAgent(unittest.TestCase):
    def test_init(self):
        params = {
            'action_dim': 16,
            'state_dim': 2120,
            'batch_size': [64, 64],
            'device': 'cpu',
            'hidden_units': [20, 20],
            'learning_rate': [0.001, 0.001],
            'loss': ['mse', 'mse'],
            'n_epochs': [1000, 1000],
            'n_layers': [2, 2]
        }
        model = SplitDynamicsModelPoseLSTM(params)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model.pkl')
        model.save(path)

        model_loaded = SplitDynamicsModelPoseLSTM.load(path)

        self.assertEqual(model.inputs, model_loaded.inputs)
        self.assertEqual(model.outputs, model_loaded.outputs)

    def test_learn(self):
        # Tested in the integrated test with Clutter Env
        pass

    def test_predict(self):
        # Tested in the integrated test with Clutter Env
        pass

    def test_clutter2array(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'splitdynamicsmodelposelstm.test_agent.test_filter_datapoint.pkl')
        with open(path, 'rb') as file:
            info = pickle.load(file)['info']
        params = {
            'action_dim': 16,
            'state_dim': 2120,
            'batch_size': [64, 64],
            'device': 'cpu',
            'hidden_units': [20, 20],
            'learning_rate': [0.001, 0.001],
            'loss': ['mse', 'mse'],
            'n_epochs': [1000, 1000],
            'n_layers': [2, 2]
        }
        model = SplitDynamicsModelPoseLSTM(params)

        haha = model._clutter2array(info)
        self.assertTrue(isinstance(haha, np.ndarray))
        self.assertEqual(haha.shape, (1002, 4))

    def test_filter_datapoint(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'splitdynamicsmodelposelstm.test_agent.test_filter_datapoint.pkl')
        with open(path, 'rb') as file:
            temp = pickle.load(file)

        info = temp['info']
        expected_result = temp['results']

        params = {
            'action_dim': 16,
            'state_dim': 2120,
            'batch_size': [64, 64],
            'device': 'cpu',
            'hidden_units': [20, 20],
            'learning_rate': [0.001, 0.001],
            'loss': ['mse', 'mse'],
            'n_epochs': [1000, 1000],
            'n_layers': [2, 2]
        }
        model = SplitDynamicsModelPoseLSTM(params)
        inputs = model._clutter2array(info)
        result = model.filter_datapoint(inputs)
        np.testing.assert_equal(result, expected_result)

        inputs2 = np.zeros(inputs.shape)
        result = model.filter_datapoint(inputs2)
        np.testing.assert_equal(result, np.zeros(inputs.shape))


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
            'hidden_units': [20, 20],
            'learning_rate': [0.001, 0.001],
            'loss': ['mse', 'mse'],
            'n_epochs': [1000, 1000],
            'n_layers': [2, 2]
        }
        torch.manual_seed(0)
        model = SplitDynamicsModelPoseLSTM(params)
        model.load_dataset(env_data)
        model.seed(0)
        model.learn()

        # Test predictions of the learned model
        # -------------------------------------
        action = 2
        env.reset()
        next_state, reward, done, info = env.step(action)
        prediction = model.predict(info['extra_data']['push_forces_vel'], action)
        np.testing.assert_equal(prediction, np.array([0.04061853885650635, 0.03152221813797951, 0.11244191974401474]))

        prediction = model.predict(info['extra_data']['push_forces_vel'], 8)
        np.testing.assert_equal(prediction, np.array([0.0, 0.0, 0.0]))

if __name__ == '__main__':
    unittest.main()
# Datapoint[x= [0.06328722 1.57079633], y= [ 0.00518818  0.05335347 -0.03774902]]
