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
            'batch_size': [64, 64],
            'device': 'cpu',
            'hidden_units': [[20], [20]],
            'learning_rate': [0.001, 0.001],
            'loss': ['mse', 'mse'],
            'nr_epochs': [1000, 1000]
        }
        model = SplitDynamicsModelPose(params)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model.pkl')
        model.save(path)

        model_loaded = SplitDynamicsModelPose.load(path)

        self.assertEqual(model.inputs, model_loaded.inputs)
        self.assertEqual(model.outputs, model_loaded.outputs)

    def test_loading(self):
        model = SplitDynamicsModelPose.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl'))
        state = 0.153253
        action = 2
        prediction = model.predict(state, action)
        expected = np.array([-0.00143145, 0.11452703, 0.00981089])
        np_test.assert_array_almost_equal(prediction, expected)

if __name__ == '__main__':
    unittest.main()
