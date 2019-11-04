import unittest
from robamine.algo.dynamicsmodel import DynamicsModel
import os
import numpy.testing as np_test
import numpy as np

class TestAgent(unittest.TestCase):
    def test_creating_a_model(self):
        params = {
            'action_dim': 16,
            'state_dim': 2120,
            'batch_size': 64,
            'device': 'cpu',
            'hidden_units': [20, 20],
            'learning_rate': 0.001,
            'loss': 'mse',
            'nr_epochs': 1000
        }
        model = DynamicsModel(params, 2, 4)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model.pkl')
        model.save(path)

        model_loaded = DynamicsModel.load(path)

        self.assertEqual(model.inputs, model_loaded.inputs)
        self.assertEqual(model.outputs, model_loaded.outputs)

if __name__ == '__main__':
    unittest.main()
