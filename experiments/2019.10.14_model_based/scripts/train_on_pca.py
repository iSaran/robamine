from robamine.algo.util import EnvData, Dataset, Datapoint, Transition
from robamine.algo.core import NetworkModel
from robamine.algo.dynamicsmodel import FullyConnectedNetwork
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch

class FCDynamicsModel(NetworkModel):
    def __init__(self, *args):
        self.pca = None
        super().__init__(*args)

    def _get_network(self):
        self.network = FullyConnectedNetwork(
            inputs=self.inputs,
            hidden_units=params['hidden_units'],
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

        self.pca = PCA(.95)
        # self.pca = PCA(n_components=30)
        pca_components = self.pca.fit_transform(data_x)
        data = Dataset.from_array(x_array=pca_components, y_array=data_y)

        # Split to train and test datasets
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

def train(agent, n_epochs):
    loss_train, loss_test  = [], []
    for i in range(n_epochs):
        train_loss, test_loss = agent.learn()
        loss_train.append(train_loss)
        loss_test.append(test_loss)
    return np.concatenate((np.array(loss_train).reshape(-1, 1), np.array(loss_test).reshape(-1, 1)), axis=1)

def get_a_push_from_clutter():
    # Create env
    params = {
      'discrete': True,
      'nr_of_actions': 16,  # u x w
      'render': True,
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
    print('handcrafted predicted_displacement:', info['extra_data']['predicted_displacement'])

    print(type(env_data.info['extra_data']))

    data = load_dataset(env_data)
    return data

if __name__ == '__main__':
    print('Loading dataset...')
    env_data_path = '/home/iason/Dropbox/projects/phd/clutter/training/2019.10.14_model_based/transitions_pose/samples.env'
    env_data = EnvData.load(env_data_path)
    dataset = load_dataset(env_data)
    dataset, eval_dataset = dataset.split(0.8) # eval dataset will be used for extracting statistics

    params = {'learning_rate' : 0.001,
              'batch_size' : 64,
              'hidden_units' : [100, 100]
              }
    print('Creating Model with params:...', params)
    model = FCDynamicsModel(params, dataset)

    print('Training Model...')
    loss = train(model, n_epochs=100)

    print('Train finished, see plotting losses...')
    plt.plot(loss)
    plt.legend(('train loss', 'test loss'))
    plt.show()

    print('Evaluating Model...')
    eval_x, eval_y = eval_dataset.to_array()
    prediction = model.predict(eval_x)
    error = prediction - eval_y
    error_x = error[:, 0]
    error_y = error[:, 1]
    error_rot = error[:, 2]
    results = [['x', np.mean(error_x), np.std(error_x), np.max(error_x)],
               ['y', np.mean(error_y), np.std(error_y), np.max(error_y)],
               ['rot', np.mean(error_rot), np.std(error_rot), np.max(error_rot)]]
    headers = ['Mean', 'Std', 'Max']

    from tabulate import tabulate
    print(tabulate(results, headers))

    # Run a random push in clutter and predict pose
    for i in range(5):
        print("Push No.: ", i)
        data = get_a_push_from_clutter()
        x, y = data.to_array()
        prediction = model.predict(x)
        print('Prediction from PCA network:', prediction)
