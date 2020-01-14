from robamine.algo.util import EnvData, Dataset, Datapoint, Transition
from robamine.algo.core import NetworkModel, SupervisedTrainWorld, Agent
from robamine.algo.dynamicsmodel import FullyConnectedNetwork
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.nn as nn
from robamine import rb_logging
import logging


# Torch Networks & support models
# -------------------------------

class AutoEncoderFC(nn.Module):
    """
    An autoencoder using FC nets. Last layer is tanh which means that it assumes
    that data are scaled in [-1, 1]
    """
    def __init__(self, init_dim, hidden_units):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(init_dim, hidden_units[0]))

        for i in range(1, len(hidden_units)):
            self.layers.append(nn.Linear(hidden_units[i - 1],
                                      hidden_units[i]))
            self.layers[-1].weight.data.uniform_(-0.003, 0.003)
            self.layers[-1].bias.data.uniform_(-0.003, 0.003)

        self.latent_layer_index = len(self.layers) - 1

        for i in range(2, len(hidden_units) + 1):
            self.layers.append(nn.Linear(hidden_units[1 - i],
                                      hidden_units[-i]))
            self.layers[-1].weight.data.uniform_(-0.003, 0.003)
            self.layers[-1].bias.data.uniform_(-0.003, 0.003)

        self.layers.append(nn.Linear(hidden_units[0], init_dim))
        print('--------------------------------------')
        print(self.layers)
        print('latent layer index:', self.latent_layer_index)
        print('--------------------------------------')

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            # x = nn.functional.relu(x)
        x = self.layers[-1](x)
        # x = torch.tanh(x)
        return x

    def latent(self, x):
        for i in range(self.latent_layer_index):
            x = self.layers[i](x)
            x = nn.functional.relu(x)
        return self.layers[self.latent_layer_index](x)

class AutoEncoderFCModel(NetworkModel):
    """
    An autoencoder using FC nets
    """
    def __init__(self, params, dataset):
        super().__init__(params=params, dataset=dataset, name='AutoEncoderFCModel')

    def _get_network(self):
        self.network = AutoEncoderFC(
            init_dim=self.inputs,
            hidden_units=self.params['hidden_units']).to(self.device)

    def _preprocess_dataset(self, dataset):
        return dataset

    def predict(self, state):
        # ndim == 1 is assumed to mean 1 sample (not multiple samples of 1 feature)
        if state.ndim == 1:
            state_ = state.reshape(1, -1)
        else:
            state_ = state

        inputs = state_.copy()
        s = torch.FloatTensor(inputs).to(self.device)
        prediction = self.network.latent(s).cpu().detach().numpy()
        return prediction


# Models for predicting pose from forces
# --------------------------------------

class PosePredictorAE(NetworkModel):
    def __init__(self, params, dataset, name='PosePredictorAE'):
        self.autoencoder = None
        self.standard_scaler_x = StandardScaler()
        self.standard_scaler_y = StandardScaler()
        self.min_max_scaler_x = MinMaxScaler(feature_range=[-1, 1])
        self.min_max_scaler_y = MinMaxScaler(feature_range=[-1, 1])
        super().__init__(params, dataset, name)

    def _get_network(self):
        self.network = FullyConnectedNetwork(
            inputs=self.inputs,
            hidden_units=self.params['hidden_units'],
            outputs=self.outputs).to(self.device)

    def _preprocess_dataset(self, dataset):
        '''Preprocesses the dataset and loads it to train and test set'''
        assert isinstance(dataset, Dataset)
        data = Dataset(dataset.copy())

        # Check and remove NaN values
        try:
            data.check()
        except ValueError:
            x, y = data.to_array()
            indexes = np.nonzero(np.isnan(y))
            for i in reversed(indexes[0]):
                del data[i]

        # Rescale
        data_x, data_y = data.to_array()
        data_x = self.standard_scaler_x.fit_transform(data_x)
        data_y = self.standard_scaler_y.fit_transform(data_y)
        # data_x = self.min_max_scaler_x.fit_transform(data_x)
        # data_y = self.min_max_scaler_y.fit_transform(data_y)

        # data_ae_x = self.min_max_scaler_x.fit_transform(data_x)
        data = Dataset.from_array(data_x, data_x)
        self.autoencoder = AutoEncoderFCModel(params = self.params['ae']['net'], dataset = data)

        SupervisedTrainWorld(agent=self.autoencoder, dataset=None,
                             params = self.params['ae']['train'],
                             name = 'fc_ae').run()

        transformed_x = self.autoencoder(data_x)
        data = Dataset.from_array(transformed_x, data_y)
        return data

    def predict(self, state):
        # ndim == 1 is assumed to mean 1 sample (not multiple samples of 1 feature)
        if state.ndim == 1:
            state_ = state.reshape(1, -1)
        else:
            state_ = state

        inputs = self.standard_scaler_x.transform(state_)
        # inputs = self.min_max_scaler_x.transform(state_)
        transformed = self.autoencoder(inputs)

        s = torch.FloatTensor(transformed).to(self.device)
        prediction = self.network(s).cpu().detach().numpy()
        # prediction = self.min_max_scaler_y.inverse_transform(prediction)[0]
        prediction = self.standard_scaler_y.inverse_transform(prediction)[0]

        return prediction

class PosePredictorPCA(NetworkModel):
    def __init__(self, params, dataset, name='PosePredictorPCA'):
        self.pca = None
        super().__init__(params, dataset, name)
        self.name = 'PosePredictorPCA'

    def _get_network(self):
        self.network = FullyConnectedNetwork(
            inputs=self.inputs,
            hidden_units=self.params['hidden_units'],
            outputs=self.outputs).to(self.device)

    def _preprocess_dataset(self, dataset):
        '''Preprocesses the dataset and loads it to train and test set'''
        assert isinstance(dataset, Dataset)
        data = Dataset(dataset.copy())
        # Check and remove NaN values
        try:
            data.check()
        except ValueError:
            x, y = data.to_array()
            indexes = np.nonzero(np.isnan(y))
            for i in reversed(indexes[0]):
                del data[i]

        data_x, data_y = data.to_array()
        data_x = self.scaler_x.fit_transform(data_x)
        # data_y = self.scaler_y.fit_transform(data_y)

        # self.pca = PCA(.95)
        self.pca = PCA(n_components=30)
        pca_components = self.pca.fit_transform(data_x)
        data = Dataset.from_array(x_array=pca_components, y_array=data_y)
        return data

    def predict(self, state):
        # ndim == 1 is assumed to mean 1 sample (not multiple samples of 1 feature)
        if state.ndim == 1:
            state_ = state.reshape(1, -1)
        else:
            state_ = state

        inputs = self.scaler_x.transform(state_)
        pca_components = self.pca.transform(inputs)

        s = torch.FloatTensor(pca_components).to(self.device)
        prediction = self.network(s).cpu().detach().numpy()
        # prediction = self.scaler_y.inverse_transform(prediction)[0]

        return prediction

class PosePredictorHandcrafted(Agent):
    def __init__(self, name='PosePredictorHandcrafted',
                 params={
                         'epsilon' : 1e-8,
                         'filter' : 0.9,
                         'outliers_cutoff' : 3.8,
                         'plot' : False
                        }):
        super().__init__(name, params)

    # def predict_displacement_from_forces(pos_measurements, force_measurements, epsilon=1e-8, filter=0.9, outliers_cutoff=3.8, plot=False):

    def predict(self, inputs):

        # Parse parameters
        epsilon = self.params['epsilon']
        filter = self.params['filter']
        outliers_cutoff = self.params['outliers_cutoff']
        plot = self.params['plot']

        import matplotlib.pyplot as plt
        from robamine.utils.math import filter_signal

        if inputs.ndim == 1:
            state = inputs.reshape(1, -1)
        else:
            state = inputs

        output = np.zeros((state.shape[0], 3))
        for datapoint in range(state.shape[0]):
            inp = state[datapoint, :].reshape(-1, 4)

            # Calculate force direction
            # -------------------------
            f = inp[:, 2:4].copy()
            f = np.nan_to_num(f / np.linalg.norm(f, axis=1).reshape(-1, 1))
            f_norm = np.linalg.norm(f, axis=1)
            # plt.plot(f)
            # plt.show()

            # Find start and end of the contacts
            first = f_norm[0]
            for i in range(f_norm.shape[0]):
                if abs(f_norm[i] - first) > epsilon:
                    break
            start_contact = i

            first = f_norm[-1]
            for i in reversed(range(f_norm.shape[0])):
                if abs(f_norm[i] - first) > epsilon:
                    break;
            end_contact = i

            # No contact with the target detected
            if start_contact > end_contact:
                continue

            f = f[start_contact:end_contact, :]

            if plot:
                fig, axs = plt.subplots(2,2)
                axs[0][0].plot(f)
                plt.title('Force')
                axs[0][1].plot(np.linalg.norm(f, axis=1))
                plt.title('norm')

            f[:,0] = filter_signal(signal=f[:,0], filter=filter, outliers_cutoff=outliers_cutoff)
            f[:,1] = filter_signal(signal=f[:,1], filter=filter, outliers_cutoff=outliers_cutoff)
            f = np.nan_to_num(f / np.linalg.norm(f, axis=1).reshape(-1, 1))

            if plot:
                axs[1][0].plot(f)
                plt.title('Filtered force')
                axs[1][1].plot(np.linalg.norm(f, axis=1))
                plt.title('norm')
                plt.show()

            # Velocity direction
            p = inp[start_contact:end_contact, :2].copy()
            p_dot = np.concatenate((np.zeros((1, 2)), np.diff(p, axis=0)))
            p_dot_norm = np.linalg.norm(p_dot, axis=1).reshape(-1, 1)
            p_dot_normalized = np.nan_to_num(p_dot / p_dot_norm)

            if plot:
                fig, axs = plt.subplots(2)
                axs[0].plot(p_dot_normalized)
                axs[0].set_title('p_dot normalized')
                axs[1].plot(p_dot)
                axs[1].set_title('p_dot')
                plt.legend(['x', 'y'])
                plt.show()

            perpedicular_to_p_dot_normalized = np.zeros(p_dot_normalized.shape)
            for i in range(p_dot_normalized.shape[0]):
                perpedicular_to_p_dot_normalized[i, :] = np.cross(np.append(p_dot_normalized[i, :], 0), np.array([0, 0, 1]))[:2]

            inner = np.diag(np.matmul(-p_dot_normalized, np.transpose(f))).copy()
            inner_perpedicular = np.diag(np.matmul(perpedicular_to_p_dot_normalized, np.transpose(f))).copy()
            if plot:
                plt.plot(inner)
                plt.title('inner product')
                plt.show()

            # Predict
            prediction = np.zeros(2)
            theta = 0.0
            for i in range(inner.shape[0]):
                prediction += p_dot_norm[i] * (inner[i] * p_dot_normalized[i, :]
                                               - inner_perpedicular[i] * perpedicular_to_p_dot_normalized[i, :])


            mean_last_inner = np.mean(inner[-10:])
            mean_last_inner = min(mean_last_inner, 1)
            mean_last_inner = max(mean_last_inner, -1)

            theta = np.sign(np.mean(inner_perpedicular[-10:])) * np.arccos(mean_last_inner)

            output[datapoint] = np.array([prediction[0], prediction[1], theta])

        return output


# General functions
# -----------------

def load_dataset(data):
    dataset = Dataset()
    for i in range(len(data.info['extra_data'])):
        primitive = int(np.floor(data.transitions[i].action / 8))

        if primitive == 0:
            pos = data.info['extra_data'][i]['push_finger_vel']
            force = data.info['extra_data'][i]['push_finger_forces']
            poses = data.info['extra_data'][i]['target_object_displacement']

            # If the following is none it means we had a collision in Clutter
            # and no useful data from pushing was recorded
            if (pos is not None) and (force is not None):
                force = np.delete(force, 2, axis=1)
                pos = np.delete(pos, 2, axis=1)
                vel = np.concatenate((np.zeros((1, 2)), np.diff(pos, axis=0)))
                inputs = np.concatenate((pos, force), axis=1).ravel().copy()
                dataset.append(Datapoint(x = inputs, y = poses[-1, :]))
    return dataset

def get_a_push_from_clutter():
    # Create env
    params = {
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
    env = gym.make('Clutter-v0', params=params)
    # env.seed(23)
    state = env.reset()
    action = 2
    next_state, reward, done, info = env.step(action)
    env_data = EnvData(['extra_data'])
    env_data.info['extra_data'].append(info['extra_data'])
    env_data.transitions.append(Transition(state, action, reward, next_state, done))
    print('real displacement:', info['extra_data']['displacement'][2])
    data = load_dataset(env_data)
    return data

def evaluate(model, eval_dataset):
    print('==================================')
    print('Evaluating Model', model.name, 'in a dataset of', len(eval_dataset), 'datapoints')
    print('==================================')
    eval_x, eval_y = eval_dataset.to_array()
    prediction = model(eval_x)
    error = np.abs(prediction - eval_y)
    error_x = error[:, 0]
    error_y = error[:, 1]
    error_rot = error[:, 2]
    results = [['x', np.mean(error_x), np.std(error_x), np.max(error_x)],
               ['y', np.mean(error_y), np.std(error_y), np.max(error_y)],
               ['rot', np.mean(error_rot), np.std(error_rot), np.max(error_rot)]]
    headers = ['Mean', 'Std', 'Max']

    from tabulate import tabulate
    print(tabulate(results, headers))


# Pipelines for different experiments
# -----------------------------------

def run():
    print('Loading dataset...')
    env_data_path = '/home/iason/Dropbox/projects/phd/clutter/training/2019.10.14_model_based/transitions_pose/samples.env'
    env_data = EnvData.load(env_data_path)
    dataset = load_dataset(env_data)
    dataset, eval_dataset = dataset.split(0.8) # eval dataset will be used for extracting statistics

    model = PosePredictorAE(params={
                                    'ae' : {
                                            'net': {
                                                    'hidden_units' : [30],
                                                    'learning_rate' : 0.001,
                                                    'batch_size' : 64
                                                   },
                                            'train': {
                                                      'epochs': 100,
                                                      'save_every': 0,
                                                     }
                                           },
                                    'learning_rate' : 0.001,
                                    'batch_size' : 64,
                                    'hidden_units' : [100, 100]
                                   },
                            dataset=dataset)

    SupervisedTrainWorld(agent=model,
                         dataset=None,
                         params = {
                                   'epochs': 100,
                                   'save_every': 0,
                                  },
                         name = 'pose_predictor_ae').run()

    model_pca = PosePredictorPCA(params={
                                    'learning_rate' : 0.001,
                                    'batch_size' : 64,
                                    'hidden_units' : [100, 100]
                                   },
                            dataset=dataset)

    SupervisedTrainWorld(agent=model_pca,
                         dataset=None,
                         params = {
                                   'epochs': 100,
                                   'save_every': 0,
                                  },
                         name = 'pose_predictor_pca').run()

    model_handcrafted = PosePredictorHandcrafted()

    evaluate(model, eval_dataset)
    evaluate(model_pca, eval_dataset)
    evaluate(model_handcrafted, eval_dataset)


    # Run a random push in clutter and predict pose
    for i in range(1):
        print("Push No.: ", i)
        data = get_a_push_from_clutter()
        x, y = data.to_array()
        prediction = model.predict(x)
        print('Prediction from ', model.name, ':', prediction)
        prediction = model_pca.predict(x)
        print('Prediction from ', model_pca.name, ':', prediction)
        prediction = model_handcrafted.predict(x)
        print('Prediction from ', model_handcrafted.name, ':', prediction)

def fit_ae():
    print('Loading and normalizing dataset...')
    env_data_path = '/home/iason/Dropbox/projects/phd/clutter/training/2019.10.14_model_based/transitions_pose/samples.env'
    env_data = EnvData.load(env_data_path)
    dataset = load_dataset(env_data)
    x, y = dataset.to_array()
    x = StandardScaler().fit_transform(x)
    x = MinMaxScaler(feature_range=[-1, 1]).fit_transform(x)
    dataset = Dataset.from_array(x, x)

    model = AutoEncoderFCModel(params = {
                                         'learning_rate' : 0.001,
                                         'batch_size' : 64,
                                         'hidden_units' : [300, 50],
                                        },
                               dataset = dataset)
    print(model.network)

    SupervisedTrainWorld(agent=model,
                         dataset=None,
                         params = {
                                   'epochs': 150,
                                   'save_every': 0,
                                  },
                         name = 'AutoencoderTrainer').run()

if __name__ == '__main__':
    rb_logging.init('/tmp/robamine_logs')
    run()
