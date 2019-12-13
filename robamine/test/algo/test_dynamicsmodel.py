import unittest
from robamine.algo.dynamicsmodel import (FCDynamicsModel, LSTMDynamicsModel,
                                         LSTMNetwork, LSTMSplitDynamicsModel,
                                         FCSplitDynamicsModel,
                                         FCAutoEncoderModel)
from robamine.algo.util import Dataset, Datapoint
import os
import numpy.testing as np_test
import numpy as np
import torch
from math import sqrt

class TestLSTMNetwork(unittest.TestCase):
    def test_creating(self):
        # Test nework
        batch_size = 32
        sequence_length = 1000
        input_dim = 4
        output_dim = 3
        network = LSTMNetwork(inputs=input_dim, hidden_dim=20, n_layers=2, outputs=output_dim)
        data = torch.randn(batch_size, sequence_length, input_dim)
        self.assertEqual(data.shape, (batch_size, sequence_length, input_dim))
        out = network(data).cpu().detach().numpy()
        self.assertEqual(out.shape, (batch_size, sequence_length, output_dim))

        # Test nework with batch size 1
        batch_size = 1
        sequence_length = 1000
        input_dim = 4
        output_dim = 3
        network = LSTMNetwork(inputs=input_dim, hidden_dim=20, n_layers=2, outputs=output_dim)
        data = torch.randn(batch_size, sequence_length, input_dim)
        self.assertEqual(data.shape, (batch_size, sequence_length, input_dim))
        out = network(data).cpu().detach().numpy()
        self.assertEqual(out.shape, (batch_size, sequence_length, output_dim))

class TestFCDynamicModel(unittest.TestCase):
    def test_init(self):
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

    def test_learn(self):
        params = {
            'device': 'cpu',
            'hidden_units': [10],
            'learning_rate': 0.1,
            'loss': 'mse',
            'batch_size': 32,
            'scaler': 'min_max'
        }

        input_dim = 4
        output_dim = 3
        n_datapoints = 2048

        dataset = Dataset()
        for i in range(n_datapoints):
            dataset.append(Datapoint(x=np.full((input_dim,), i), y=np.full((output_dim,), sqrt(i))))

        # TODO: The order for properly seeding the learning procedure is a bit
        # weird: The model should be seeded after loading the dataset to ensure
        # correct seeding of minibatching. Fix this.
        torch.manual_seed(0)
        model = FCDynamicsModel(params, input_dim, output_dim)
        model.load_dataset(dataset)
        model.seed(0)
        for i in range(100):
            model.learn()

        state = np.full((input_dim,), 100)
        prediction = model.predict(state)
        self.assertEqual(float(model.info['train']['loss']), 0.0009492546087130904)
        np.testing.assert_equal(prediction, np.array([10.195260047912598, 10.19526195526123, 10.195259094238281]))

class TestFCAutoEncoderModel(unittest.TestCase):
    def test_init(self):
        params = {
            'batch_size': 64,
            'device': 'cpu',
            'learning_rate': 0.001,
            'loss': 'mse',
            'n_epochs': 100,
            'latent_dim': 32
        }
        model = FCAutoEncoderModel(params, 128)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model.pkl')
        model.save(path)

        model_loaded = FCAutoEncoderModel.load(path)

        self.assertEqual(model.inputs, model_loaded.inputs)
        self.assertEqual(model.outputs, model_loaded.outputs)

    def test_learn(self):
        params = {
            'batch_size': 64,
            'device': 'cpu',
            'learning_rate': 0.001,
            'loss': 'mse',
            'n_epochs': 100,
            'latent_dim': 32
        }

        input_dim = 128
        n_datapoints = 2048

        dataset = Dataset()
        for i in range(n_datapoints):
            dataset.append(Datapoint(x=np.full((input_dim,), i), y=None))

        # TODO: The order for properly seeding the learning procedure is a bit
        # weird: The model should be seeded after loading the dataset to ensure
        # correct seeding of minibatching. Fix this.
        torch.manual_seed(0)
        model = FCAutoEncoderModel(params, input_dim)
        model.load_dataset(dataset)
        model.seed(0)
        for i in range(100):
            model.learn()

        state = np.full((input_dim,), 100)
        prediction = model.predict(state)
        self.assertEqual(float(model.info['train']['loss']), 0.7252576351165771)
        self.assertEqual(prediction.shape, (1, 32))

class TestLSTMDynamicsModel(unittest.TestCase):
    def test_init(self):
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

    def test_learn(self):
        params = {
            'device': 'cpu',
            'hidden_units': 10,
            'learning_rate': 0.01,
            'loss': 'mse',
            'n_layers': 2,
            'batch_size': 32,
            'scaler': 'min_max'
        }

        sequence_length = 2
        input_dim = 4
        output_dim = 3
        n_datapoints = 2048

        dataset = Dataset()
        for i in range(n_datapoints):
            dataset.append(Datapoint(x=np.full((sequence_length, input_dim), i), y=np.full((sequence_length, output_dim), sqrt(sequence_length * i))))

        # TODO: The order for properly seeding the learning procedure is a bit
        # weird: The model should be seeded after loading the dataset to ensure
        # correct seeding of minibatching. Fix this.
        torch.manual_seed(0)
        model = LSTMDynamicsModel(params, input_dim, output_dim)
        model.load_dataset(dataset)
        model.seed(0)

        train_loss_expected = [0.16893060505390167,
                               0.025982027873396873,
                               0.005653527099639177,
                               0.003036701586097479,
                               0.004060985986143351,
                               0.001505931606516242,
                               0.0011766261886805296,
                               0.0012346295407041907,
                               0.0007550095324404538,
                               0.0006367720779962838]

        test_loss_expected = [0.7156387567520142,
                              0.1484566330909729,
                              0.07711506634950638,
                              0.05929327383637428,
                              0.03740023076534271,
                              0.03175796940922737,
                              0.026821406558156013,
                              0.02212144061923027,
                              0.02064170502126217,
                              0.023261982947587967]

        for i in range(10):
            model.learn()
            self.assertEqual(float(model.info['train']['loss']), train_loss_expected[i])
            self.assertEqual(float(model.info['test']['loss']), test_loss_expected[i])

        state = np.full((sequence_length, input_dim), 100)
        prediction = model.predict(state)
        np.testing.assert_equal(prediction, np.array([13.999224662780762, 14.00013256072998, 13.998896598815918]))

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
