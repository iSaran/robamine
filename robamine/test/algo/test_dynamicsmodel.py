import unittest
from robamine.algo.dynamicsmodel import FCDynamicsModel, LSTMDynamicsModel, LSTMNetwork, LSTMSplitDynamicsModel, FCSplitDynamicsModel
import os
import numpy.testing as np_test
import numpy as np
import torch

class TestNetworks(unittest.TestCase):
    def test_creating(self):
        # Test nework
        network = LSTMNetwork(inputs=4, hidden_dim=10, n_layers=2, outputs=3)
        batch_size = 1
        sequence_length = 10
        input_dim = 4
        data = torch.randn(batch_size, sequence_length, input_dim)
        out = network(data).cpu().detach().numpy()

class TestDynamicModels(unittest.TestCase):
    def test_creating(self):
        params = {
            'action_dim': 16,
            'state_dim': 2120,
            'batch_size': 64,
            'device': 'cpu',
            'hidden_units': [20, 20],
            'learning_rate': 0.001,
            'loss': 'mse',
            'n_epochs': 1000
        }
        model = FCDynamicsModel(params, 2, 4)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model.pkl')
        model.save(path)

        model_loaded = FCDynamicsModel.load(path)

        self.assertEqual(model.inputs, model_loaded.inputs)
        self.assertEqual(model.outputs, model_loaded.outputs)

        params = {
            'device': 'cpu',
            'hidden_units': 10,
            'learning_rate': 0.001,
            'loss': 'mse',
            'n_layers': 2
        }
        model = LSTMDynamicsModel(params, 4, 3)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model.pkl')
        model.save(path)

        model_loaded = LSTMDynamicsModel.load(path)

        self.assertEqual(model.inputs, model_loaded.inputs)
        self.assertEqual(model.outputs, model_loaded.outputs)


class TestSplitDynamicModels(unittest.TestCase):
    def test_creating(self):
        params = {
            'action_dim': 16,
            'state_dim': 2120,
            'device': 'cpu',
            'hidden_units': [10, 10],
            'learning_rate': [0.001, 0.001],
            'loss': ['mse', 'mse'],
            'n_layers': [2, 2],
            'batch_size': [64, 64],
            'n_epochs': [1000, 1000]
        }
        model = LSTMSplitDynamicsModel(params=params, inputs=4, outputs=3)

        params = {
            'action_dim': 16,
            'state_dim': 2120,
            'device': 'cpu',
            'hidden_units': [[10], [10]],
            'learning_rate': [0.001, 0.001],
            'loss': ['mse', 'mse'],
            'batch_size': [64, 64],
            'n_epochs': [1000, 1000]
        }
        model = FCSplitDynamicsModel(params=params, inputs=4, outputs=3)

if __name__ == '__main__':
    unittest.main()
