import unittest
from robamine.algo.dynamicsmodel import FCDynamicsModel, LSTMDynamicsModel, LSTMNetwork, LSTMSplitDynamicsModel, FCSplitDynamicsModel
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
        self.assertEqual(out.shape, (batch_size, output_dim))

        # Test nework with batch size 1
        batch_size = 1
        sequence_length = 1000
        input_dim = 4
        output_dim = 3
        network = LSTMNetwork(inputs=input_dim, hidden_dim=20, n_layers=2, outputs=output_dim)
        data = torch.randn(batch_size, sequence_length, input_dim)
        self.assertEqual(data.shape, (batch_size, sequence_length, input_dim))
        out = network(data).cpu().detach().numpy()
        self.assertEqual(out.shape, (batch_size, output_dim))

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
            'batch_size': 32
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
        model.load_dataset(dataset, rescale=True)
        model.seed(0)
        for i in range(100):
            model.learn()

        state = np.full((input_dim,), 100)
        prediction = model.predict(state)
        self.assertEqual(float(model.info['train']['loss']), 0.0009492546087130904)
        np.testing.assert_equal(prediction, np.array([10.195260047912598, 10.19526195526123, 10.195259094238281]))

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
            'batch_size': 32
        }

        sequence_length = 2
        input_dim = 4
        output_dim = 3
        n_datapoints = 2048

        dataset = Dataset()
        for i in range(n_datapoints):
            dataset.append(Datapoint(x=np.full((sequence_length, input_dim), i), y=np.full((output_dim,), sqrt(sequence_length * i))))

        # TODO: The order for properly seeding the learning procedure is a bit
        # weird: The model should be seeded after loading the dataset to ensure
        # correct seeding of minibatching. Fix this.
        torch.manual_seed(0)
        model = LSTMDynamicsModel(params, input_dim, output_dim)
        model.load_dataset(dataset, rescale=True)
        model.seed(0)

        train_loss_expected = [0.168924018740654,
                               0.010207300074398518,
                               0.005141713190823793,
                               0.002354690106585622,
                               0.0040090493857860565,
                               0.0007962704403325915,
                               0.0008550113416276872,
                               0.0006611660937778652,
                               0.0005286139203235507,
                               0.00043144053779542446]

        test_loss_expected = [0.7156064510345459,
                              0.127550408244133,
                              0.046276867389678955,
                              0.029462555423378944,
                              0.020332880318164825,
                              0.024569299072027206,
                              0.025565791875123978,
                              0.023401683196425438,
                              0.020883828401565552,
                              0.02272590808570385]

        for i in range(10):
            model.learn()
            self.assertEqual(float(model.info['train']['loss']), train_loss_expected[i])
            self.assertEqual(float(model.info['test']['loss']), test_loss_expected[i])

        state = np.full((sequence_length, input_dim), 100)
        prediction = model.predict(state)
        np.testing.assert_equal(prediction, np.array([13.625250816345215, 13.623515129089355, 13.627643585205078]))

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
