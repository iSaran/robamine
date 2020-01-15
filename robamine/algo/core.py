"""
Core
====

This module contains the core classes of the package for defining basic
interfaces. These base classes can be extended for implementing different RL
algorithms. Currently, the base classes for an RL agent are defined and for Neural Network in order to combine it with Deep Learning.
"""

import gym
import tensorflow as tf
from robamine.algo.util import (DataStream, Stats, get_now_timestamp,
                                print_progress, Plotter, get_agent_handle,
                                transform_sec_to_timestamp, Transition, EnvData,
                                TimestepData, EpisodeData, EpisodeListData,
                                Dataset, Datapoint)
from robamine.utils.info import get_pc_and_version, get_dir_size, get_now_timestamp
from robamine.utils.memory import ReplayBuffer
from robamine import rb_logging
import logging
import os
import pickle
from enum import Enum
import time
import datetime
import numpy as np
import yaml
from threading import Lock

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shutil

logger = logging.getLogger('robamine.algo.core')

class Agent:
    def __init__(self, name='', params={}):
        self.name = name
        self.params = params.copy()
        self.info = {}
        self.rng = np.random.RandomState()
        self.prediction_horizon = params.get('prediction_horizon', None)
        logger.debug('Agent:' + self.name + ': Created with params:' + str(params))

    def state_dict(self):
        state_dict = {}
        state_dict['name'] = self.name
        state_dict['params'] = self.params.copy()
        return state_dict

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.state_dict(), file)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as file:
            state_dict = pickle.load(file)
        self = cls.load_state_dict(state_dict)
        return self

    @classmethod
    def load_state_dict(cls, state_dict):
        raise NotImplementedError()

    def load_trainable(self):
        error = 'Agent ' + self.name + ' does not provide loading capabilities.'
        logger.error(error)
        raise NotImplementedError(error)

    def seed(self, seed=None):
        self.rng.seed(seed)

    def learn(self, state, action, reward, next_state, done):
        """
        Implements the RL algorithm which optimizes the policy by the given
        experience in each timestep. It is used by :meth:`.train`.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment
        action : np.ndarray
            The action performed by the exploration policy.
        reward : float
            The reward given by the environment upon applying the action
        next_state : np.ndarray
            The next state in which the environment transitioned after applying
            the action
        terminal : float
            1 if the next state is a terminal state, 0 otherwise.
        """
        error = 'Agent ' + self.name + ' does not implemement a learning algorithm.'
        logger.error(error)
        raise NotImplementedError(error)

    def predict(self, state):
        """
        Represents the learned policy which provides optimal actions. It is used from :meth:`.evaluate`

        Parameters
        ----------

        state : numpy array
            The current state of the environment.

        Returns
        -------
        numpy array:
            The optimal action to be performed.
        """
        error = 'Agent ' + self.name + ' does not implement a policy.'
        logger.error(error)
        raise NotImplementedError(error)

    def load_dataset(self):
        pass

    def __call__(self, input):
        return self.predict(input)

class RLAgent(Agent):
    """
    Base class for creating an RL agent.

    Example
    -------
    Assume you want to implement the Best RL Algorithm (BRLA). Create a class
    which inherits from :class:`.Agent` and implement its member methods. Then,
    you should be able to train this agent for 1 million episodes in the
    ``MyRobot`` environment as simple as:

    .. code-block:: python

        with tf.Session as session:
            agent = BRLA(session, 'MyRobot').train(1e6)

    Then you should be able to evaluate and see what the agent learned by:

    .. code-block:: python

        agent.evaluate()

    Parameters
    ----------
    sess : :class:`.tf.Session`
    env : str
        A string with the name of a registered Gym Environment
    random_seed : int, optional
        A random seed for reproducable results.
    log_dir : str
        A directory for storing the trained model and logged data.
    name : str, optional
        A name for the agent.
    """

    def __init__(self, state_dim, action_dim, name='', params={}):
        super(RLAgent, self).__init__(name, params)
        self.state_dim, self.action_dim  = state_dim, action_dim

    def explore(self, state):
        """
        Represents the exploration policy. It is used by :meth:`.train` to
        produce explorative actions. In the general case the policy for
        exploration may be different from the learned policy that we want to
        optimize (and thus the :meth:.`explore` and the :meth:.`evaluate`
        functions may be different). If you do not implement it it will sample
        the action space uniformly.

        Parameters
        ----------

        state : numpy array
            The current state of the environment.

        Returns
        -------
        numpy array
            An action to be performed for exploration.
        """
        error = 'Agent ' + self.name + ' does not implement an exploration policy.'
        logger.error(error)
        raise NotImplementedError(error)

    def q_value(self, state, action):
        error = 'Agent ' + self.name + ' does not provide a Q value.'
        logger.error(error)
        raise NotImplementedError(error)

class Network:
    """
    Base class for creating a Neural Network. During the construction of an object the :meth:`.create_architecture` is called.

    Attributes
    ----------
    inputs : tf.Tensor
        A Tensor representing the input layer of the network.
    out : tf.Tensor
        A Tensor representing the output layer of the network.
    net_params : list of tf.Variable
        The network learnable parameters.
    input_dim : int
        The dimensions of the input layer
    hidden_dims : tuple
        The dimensions of each hidden layer. The size of tuple defines the
        number of the hidden layers.
    out_dim : int
        The dimensions of the output layer
    name : str, optional
        The name of the Neural Network

    Parameters
    ----------
    sess : :class:`.tf.Session`
    input_dim : int
        The dimensions of the input layer
    hidden_dims : tuple
        The dimensions of each hidden layer. The size of tuple defines the
        number of the hidden layers.
    out_dim : int
        The dimensions of the output layer
    name : str, optional
        The name of the Neural Network
    """

    def __init__(self, sess, input_dim, hidden_units, output_dim, name):
        self.sess, self.input_dim, self.hidden_units, self.output_dim, self.name = \
                sess, input_dim, hidden_units, output_dim, name

        self.input, self.out, self.net_params, self.trainable = None, None, None, None

    @classmethod
    def create(cls):
        raise NotImplementedError

    def predict(self, inputs):
        """
        Predicts a new output given some input.

        Parameters
        ----------
        inputs : tf.Tensor
            The input values for prediction.

        Returns
        -------
        tf.Tensor
            A Tensor with the predicted output values.
        """
        raise NotImplementedError

    def learn(self, inputs, output):
        """
        Trains the neural network.

        Parameters
        ----------
        inputs : tf.Tensor
            The input dataset.
        output : tf.Tensor
            Other variable used for training. In the simple case could be output dataset. In more advance cases could be any variable for a Tensorflow placeholder that can be used for custom Tensorflow operations for training.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the child class.
        """
        raise NotImplementedError

    def get_params(self, name = None):
        if name is None:
            return self.sess.run(self.net_params)
        else:
            k = [v for v in self.net_params if v.name == name][0]
            return self.sess.run(k)

    @classmethod
    def load(cls, sess, input_dim, hidden_units, output_dim, name, trainable):
        self = cls.create(sess, input_dim, hidden_units, output_dim, name)
        sess.run([self.net_params[i].assign(trainable[i]) for i in range(len(self.net_params))])
        return self

class NetworkModel(Agent):
    '''
    A class for train PyTorch networks. Inherit and create a self.network (which
    inherits from torch.nn.Module) before calling super().__init__()
    '''
    def __init__(self, params, dataset, name='NetworkModel'):
        super().__init__(name=name, params=params)
        self.device = self.params.get('device', 'cpu')

        # Set up scalers
        self.scaler = self.params.get('scaler', 'standard')
        if self.scaler == 'min_max':
            self.range = [-1, 1]
            self.scaler_x = MinMaxScaler(feature_range=self.range)
            self.scaler_y = MinMaxScaler(feature_range=self.range)
        elif self.scaler == 'standard':
            self.range = [-1, 1]
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
        elif self.scaler is None:
            self.scaler_x = None
            self.scaler_y = None
        else:
            raise ValueError(self.name + ': Select scaler between: None, min_max, standard.')

        # Preprocess dataset and extract inputs and outputs
        self.train_dataset, self.test_dataset = self._preprocess_dataset(dataset).split(0.7)
        assert isinstance(self.train_dataset, Dataset)
        assert isinstance(self.test_dataset, Dataset)
        self.inputs = self.train_dataset[0].x.shape[0]
        self.outputs = self.train_dataset[0].y.shape[0]

        self._get_network()

        # Create the networks, optimizers and loss
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=self.params['learning_rate'])

        loss = self.params.get('loss', 'mse')
        if loss == 'mse':
            self.loss = nn.MSELoss()
        elif loss == 'huber':
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError('DynamicsModel: Loss should be mse or huber')

        self.iterations = 0
        self.info['train'] = {'loss': 0.0}
        self.info['test'] = {'loss': 0.0}

    def _get_network(self):
        "Write a nn.Module object to self.network"
        raise NotImplementedError()

    def _preprocess_dataset(self, dataset):
        '''Preprocesses the dataset and loads it to train and test set'''
        assert isinstance(dataset, Dataset)
        data = dataset.copy()

        # Check and remove NaN values
        try:
            data.check()
        except ValueError:
            x, y = data.to_array()
            indexes = np.nonzero(np.isnan(y))
            for i in reversed(indexes[0]):
                del data[i]

        # Rescale
        if self.scaler:
            data_x, data_y = data.to_array()
            data_x = self.scaler_x.fit_transform(data_x)
            data_y = self.scaler_y.fit_transform(data_y)
            data = Dataset.from_array(data_x, data_y)

        return data

    def predict(self, state):
        # ndim == 1 is assumed to mean 1 sample (not multiple samples of 1 feature)
        if state.ndim == 1:
            state_ = state.reshape(1, -1)
        else:
            state_ = state

        if self.scaler:
            inputs = self.scaler_x.transform(state_)
        else:
            inputs = state_.copy()
        s = torch.FloatTensor(inputs).to(self.device)
        prediction = self.network(s).cpu().detach().numpy()
        if self.scaler:
            prediction = self.scaler_y.inverse_transform(prediction)[0]

        return prediction

    def learn(self):
        '''Run one epoch'''
        self.iterations += 1

        # Calculate loss in train dataset
        train_x, train_y = self.train_dataset.to_array()
        real_x = torch.FloatTensor(train_x).to(self.device)
        prediction = self.network(real_x)
        real_y = torch.FloatTensor(train_y).to(self.device)
        loss = self.loss(prediction, real_y)
        self.info['train']['loss'] = loss.detach().cpu().numpy().copy()

        # Calculate loss in test dataset
        test_x, test_y = self.test_dataset.to_array()
        real_x = torch.FloatTensor(test_x).to(self.device)
        prediction = self.network(real_x)
        real_y = torch.FloatTensor(test_y).to(self.device)
        loss = self.loss(prediction, real_y)
        self.info['test']['loss'] = loss.detach().cpu().numpy().copy()

        # Minimbatch update of network
        minibatches = self.train_dataset.to_minibatches(
            self.params['batch_size'])
        for minibatch in minibatches:
            batch_x, batch_y = minibatch.to_array()

            real_x = torch.FloatTensor(batch_x).to(self.device)
            prediction = self.network(real_x)
            real_y = torch.FloatTensor(batch_y).to(self.device)
            loss = self.loss(prediction, real_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self.info['train']['loss'], self.info['test']['loss']

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['trainable'] = self.trainable_dict()
        state_dict['iterations'] = self.iterations
        state_dict['scaler_x'] = self.scaler_x
        state_dict['scaler_y'] = self.scaler_y
        state_dict['inputs'] = self.inputs
        state_dict['outputs'] = self.outputs
        return state_dict

    def trainable_dict(self):
        return self.network.state_dict()

    def load_trainable_dict(self, trainable):
        self.network.load_state_dict(trainable)

    def load_trainable(self, file_path):
        '''Assume that file path is a pickle with with self.state_dict() '''
        state_dict = pickle.load(open(input, 'rb'))
        self.load_trainable_dict(state_dict['trainable'])

    @classmethod
    def load_state_dict(cls, state_dict):
        self = cls(state_dict['params'], state_dict['inputs'],
                   state_dict['outputs'])
        self.load_trainable_dict(state_dict['trainable'])
        self.iterations = state_dict['iterations']
        self.scaler_x = state_dict['scaler_x']
        self.scaler_y = state_dict['scaler_y']
        return self

    def seed(self, seed=None):
        super().seed(seed)
        self.train_dataset.seed(seed)
        self.test_dataset.seed(seed)

# World classes

class WorldState(Enum):
    IDLE = 1
    RUNNING = 2
    FINISHED = 3

class World:
    def __init__(self, name=None):
        if name is None:
            self.name = 'world_' + get_now_timestamp()
        else:
            self.name = name

        self.stop_running = False
        self.results_lock = Lock()
        self.state_lock = Lock()
        self.current_state = WorldState.IDLE

        # Setup logging directory and tf writers
        self.log_dir = os.path.join(rb_logging.get_logger_path(), self.name.lower())


        if os.path.exists(self.log_dir):
            logger.warn("Removing existing world directory. This may happen if you set the same name for different worlds")
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir)
        self.sess = tf.Session()
        self.tf_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.stats = None

        self.config = {}
        self.config['results'] = {}
        self.config['results']['logging_dir'] = self.log_dir
        hostname, username, version = get_pc_and_version()
        self.config['results']['hostname'] = hostname + ':' + username
        self.config['results']['version'] = version
        self.config['results']['started_on'] = None
        self.config['results']['estimated_time'] = None
        self.config['results']['time_elapsed'] = None
        self.config['results']['dir_size'] = 0
        self.config['results']['progress'] = 0.0

        logger.info('World:' + self.name + ': Created.')

    def reset(self):
        self.stop_running = False
        self.set_state(WorldState.IDLE)
        self.start_time = time.time()
        self.config['results']['started_on'] = str(datetime.datetime.now())

    def set_state(self, state):
        self.state_lock.acquire()
        self.current_state = state
        self.state_lock.release()

    def get_state(self):
        self.state_lock.acquire()
        result = self.current_state
        self.state_lock.release()
        return result

    def run(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def update_results(self, progress, thread_safe=True, write_yaml=True):

        # Update the results fields in config
        if thread_safe:
            self.results_lock.acquire()

        self.config['results']['progress'] = progress
        self.config['results']['time_elapsed'] = transform_sec_to_timestamp(time.time() - self.start_time)
        self.config['results']['estimated_time'] = transform_sec_to_timestamp(((1 - progress) * (time.time() - self.start_time)) / progress)
        self.config['results']['dir_size'] = get_dir_size(self.log_dir)

        if thread_safe:
            self.results_lock.release()

        # Update YAML file
        if write_yaml:
            with open(os.path.join(self.log_dir, 'config.yml'), 'w') as outfile:
                yaml.dump(self.config, outfile, default_flow_style=False)

class SupervisedTrainWorld(World):
    def __init__(self, agent, dataset, params, name=None):
        super(SupervisedTrainWorld, self).__init__(name)
        self.epochs = params.get('epochs', 0)
        self.save_every = params.get('save_every', 0)
        if dataset:
            self.dataset = pickle.load(open(dataset, 'rb'))

        # Agent setup
        if isinstance(agent, str):
            agent_name = agent
            agent_handle = get_agent_handle(agent_name)
        elif isinstance(agent, dict):
            agent_name = agent['name']
            print(agent['name'])
            agent_handle = get_agent_handle(agent_name)
            agent_params = agent['params'] if 'params' in agent else {}
            self.agent = agent_handle(params = agent_params)
        elif isinstance(agent, Agent):
            self.agent = agent
        else:
            err = ValueError('Provide an Agent or a string in order to create an agent for the new world')
            logger.exception(err)
            raise err
        self.agent_name = self.agent.name

        if dataset:
            self.agent.load_dataset(self.dataset)

        # Create datastreams
        self.datastream = {}
        self.groups = list(self.agent.info.keys())
        self.variables = {}
        for group in self.groups:
            self.variables[group] = list(self.agent.info[group].keys())
            self.datastream[group] = DataStream(self.sess, self.log_dir, \
                                                self.tf_writer, \
                                                self.variables[group], \
                                                group)

        # Setup the internal config dictionary
        self.config['results']['n_epochs'] = 0

    def run(self):
        logger.info('SupervisedTrainWorld: %s: running: %s for %d epochs', self.name, self.agent_name, self.epochs)
        self.reset()
        self.set_state(WorldState.RUNNING)
        for i in range(self.epochs):
            self.agent.learn()

            if self.stop_running:
                break

            # Log agent's info
            for group in self.groups:
                row = []
                for var in self.variables[group]:
                    row.append(self.agent.info[group][var])
                self.datastream[group].log(i, row)


            # Save agent model
            self.save()
            if self.save_every and (i + 1) % self.save_every == 0:
                self.save(suffix='_' + str(i+1))

            # Update results
            self.update_results(n_epochs=i+1)

        self.set_state(WorldState.FINISHED)

    def reset(self):
        super(SupervisedTrainWorld, self).reset()
        self.config['results']['n_epochs'] = 0

    @classmethod
    def from_dict(cls, config):
        self = cls(agent=config['agent'],
                   dataset=config['env']['params']['path'], \
                   epochs=config['world']['params']['epochs'], \
                   save_every=config['world']['params']['save_every'])

        if 'trainable_params' in config['agent'] and config['agent']['trainable_params'] != '':
            self.agent.load_trainable(config['agent']['trainable_params'])

        # Save the config
        self.config['world'] = config['world'].copy()
        self.config['env'] = config['env'].copy()
        self.config['agent'] = config['agent'].copy()

        return self

    def update_results(self, n_epochs, thread_safe=True, write_yaml=True):

        # Update the results fields in config
        if thread_safe:
            self.results_lock.acquire()

        self.config['results']['n_epochs'] = n_epochs
        prog = self.config['results']['n_epochs'] / self.epochs

        super(SupervisedTrainWorld, self).update_results(progress=prog, thread_safe=False, write_yaml=False)

        if thread_safe:
            self.results_lock.release()

        # Update YAML file
        if write_yaml:
            with open(os.path.join(self.log_dir, 'config.yml'), 'w') as outfile:
                yaml.dump(self.config, outfile, default_flow_style=False)

    def save(self, suffix=''):
        agent_model_file_name = os.path.join(self.log_dir, 'model' + suffix + '.pkl')
        self.agent.save(agent_model_file_name)

class RLWorld(World):
    def __init__(self, agent, env, params, name=None):
        super(RLWorld, self).__init__(name)
        # Environment setup
        if isinstance(env, gym.Env):
            self.env = env
        elif isinstance(env, str):
            self.env = gym.make(env)
            if isinstance(self.env.observation_space, gym.spaces.dict.Dict):
                logger.warn('Gym environment has a %s observation space. I will wrap it with a gym.wrappers.FlattenDictWrapper.', type(self.env.observation_space))
                self.env = gym.wrappers.FlattenDictWrapper(self.env, ['observation', 'desired_goal'])
            self.config['env'] = {}
            self.config['env']['name'] = agent
        elif isinstance(env, dict):
            env_params = env['params'] if 'params' in env else {}
            self.config['env'] = env
            self.env = gym.make(env['name'], params=env_params)
        else:
            err = ValueError('Provide a gym.Env or a string in order to create a new world')
            logger.exception(err)
            raise err

        # State dimensions
        self.state_dim = int(self.env.observation_space.shape[0])
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = int(self.env.action_space.shape[0])
        self.env_name = self.env.spec.id

        # Agent setup
        if isinstance(agent, str):
            agent_name = agent
            agent_handle = get_agent_handle(agent_name)
            if (agent_name == 'Dummy'):
                self.agent = agent_handle(self.env.action_space, self.state_dim, self.action_dim)
            else:
                self.agent = agent_handle(self.state_dim, self.action_dim)
            self.config['agent'] = {}
            self.config['agent']['name'] = agent
        elif isinstance(agent, dict):
            agent_name = agent['name']
            agent_handle = get_agent_handle(agent_name)
            agent_params = agent['params'] if 'params' in agent else {}
            self.agent = agent_handle(self.state_dim, self.action_dim, agent_params)
            self.config['agent'] = agent
        elif isinstance(agent, Agent):
            self.agent = agent
        else:
            err = ValueError('Provide an Agent or a string in order to create an agent for the new world')
            logger.exception(err)
            raise err
        self.agent_name = self.agent.name

        # Check if environment and agent are compatible
        try:
            assert self.agent.state_dim == self.state_dim, 'Agent and environment has incompatible state dimension'
            assert self.agent.action_dim == self.action_dim, 'Agent and environment has incompantible action dimension'
        except AssertionError as err:
            logger.exception(err)
            raise err

        # Setup the internal config dictionary
        self.config['results']['n_episodes'] = 0
        self.config['results']['n_timesteps'] = 0

        logger.info('Initialized world with the %s in the %s environment', self.agent_name, self.env.spec.id)

        self.iteration_name = 'episodes'
        self.iterations = params.get(self.iteration_name, 0)
        self.save_every = params.get('save_every', None)
        self.render = params.get('render', False)

        # Other variables for running episodes
        self.experience_time = 0.0
        self.start_time = None

        # A list in which dictionaries for episodes stats are stored
        self.episode_stats = []

        if 'List of env init states' in params and params['List of env init states'] != '':
            env_data = EnvData.load(params['List of env init states'])
            self.env_init_states = env_data.init_states
        else:
            self.env_init_states = None

        self.episode_list_data = EpisodeListData()

    @classmethod
    def from_dict(cls, config):
        # Setup the environment
        if len(config['env']) == 1:
            env = gym.make(config['env']['name'] + '-v0')
        else:
            env = gym.make(config['env']['name'] + '-v0', params=config['env']['params'])
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            logger.warn('Gym environment has a %s observation space. I will wrap it with a gym.wrappers.FlattenDictWrapper.', type(env.observation_space))
            env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'desired_goal'])

        state_dim = int(env.observation_space.shape[0])
        if isinstance(env.action_space, gym.spaces.discrete.Discrete):
            action_dim = env.action_space.n
        else:
            action_dim = int(env.action_space.shape[0])

        # Create the world
        self = cls(config['agent'], env, config['world']['params'], config['world']['name'])

        if 'trainable_params' in config['agent'] and config['agent']['trainable_params'] != '':
            self.agent.load_trainable(config['agent']['trainable_params'])

        # Save the config
        self.config['world'] = config['world'].copy()
        self.config['env'] = config['env'].copy()
        self.config['agent'] = config['agent'].copy()

        return self

    def seed(self, seed):
        self.env.seed(seed)
        self.agent.seed(seed)

    def reset(self):
        super(RLWorld, self).reset()
        self.experience_time = 0
        self.episode_stats = []
        self.episode_list_data = EpisodeListData()
        self.config['results']['n_episodes'] = 0
        self.config['results']['n_timesteps'] = 0

    def run_episode(self, episode, i):
        # Run the episode. Assumed that the episode has been already created by
        # child classes

        if self.env_init_states:
            init_state = self.env_init_states[i]
        else:
            init_state = None

        episode.run(render=self.render, init_state=init_state)

        # Update tensorboard stats
        self.stats.update(i, episode.stats)
        self.episode_stats.append(episode.stats)
        self.episode_list_data.append(episode.data)

        # Save agent model
        self.save()
        if self.save_every and (i + 1) % self.save_every == 0:
            self.save(suffix='_' + str(i+1))

        # Save the config in YAML file
        self.experience_time += episode.stats['experience_time']
        self.update_results(n_iterations = i + 1, n_timesteps = episode.stats['n_timesteps'])

    def run(self):
        logger.info('%s running on %s for %d episodes', self.agent_name, self.env_name, self.iterations)
        self.reset()
        self.set_state(WorldState.RUNNING)
        for i in range(self.iterations):
            self.run_episode(i)

            if self.stop_running:
                break

        self.set_state(WorldState.FINISHED)

    def plot(self, batch_size=5):
        if self.train_stats:
            Plotter.create_batch_from_stream(self.log_dir, 'train', batch_size)
            plotter = Plotter(self.log_dir, ['train', 'batch_train'])
            plotter.plot()
        if self.eval_stats:
            Plotter.create_batch_from_stream(self.log_dir, 'eval', batch_size)
            plotter = Plotter(self.log_dir, ['eval', 'batch_train'])
            plotter.plot()

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, 'config.yml'), 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        if len(config['env']) == 1:
            env = gym.make(config['env']['name'])
        else:
            env = gym.make(config['env']['name'], params=config['env'])
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            logger.warn('Gym environment has a %s observation space. I will wrap it with a gym.wrappers.FlattenDictWrapper.', type(env.observation_space))
            env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'desired_goal'])

        agent_name = config['agent']['name']
        agent_handle = get_agent_handle(agent_name)
        agent = agent_handle.load(os.path.join(directory, 'model.pkl'))

        self = cls(agent, env)
        logger.info('World loaded from %s', directory)
        return self

    def save(self, suffix=''):
        pickle.dump(self.episode_stats, open(os.path.join(self.log_dir, 'episode_stats.pkl'), 'wb'))
        self.episode_list_data.save(self.log_dir)

    def update_results(self, n_iterations, n_timesteps, thread_safe=True, write_yaml=True):

        # Update the results fields in config
        if thread_safe:
            self.results_lock.acquire()

        self.config['results']['n_' + self.iteration_name] = n_iterations
        self.config['results']['n_timesteps'] += n_timesteps
        prog = self.config['results']['n_' + self.iteration_name] / self.iterations
        super(RLWorld, self).update_results(progress=prog, thread_safe=False, write_yaml=False)

        if thread_safe:
            self.results_lock.release()

        # Update YAML file
        if write_yaml:
            with open(os.path.join(self.log_dir, 'config.yml'), 'w') as outfile:
                yaml.dump(self.config, outfile, default_flow_style=False)

class TrainWorld(RLWorld):
    def __init__(self, agent, env, params, name=None):
        super(TrainWorld, self).__init__(agent, env, params, name)
        self.stats = Stats(self.sess, self.log_dir, self.tf_writer, 'train', self.agent.info)

    def run_episode(self, i):
        episode = TrainingEpisode(self.agent, self.env)
        super(TrainWorld, self).run_episode(episode, i)

    def save(self, suffix=''):
        super(TrainWorld, self).save(suffix)
        agent_model_file_name = os.path.join(self.log_dir, 'model' + suffix + '.pkl')
        self.agent.save(agent_model_file_name)

class EvalWorld(RLWorld):
    def __init__(self, agent, env, params, name=''):
        super(EvalWorld, self).__init__(agent, env, params, name)
        self.stats = Stats(self.sess, self.log_dir, self.tf_writer, 'eval', self.agent.info)
        self.expected_values_file = open(os.path.join(self.log_dir, 'expected_values' + '.csv'), "w+")
        self.expected_values_file.write('expected,real\n')
        self.actions_file = open(os.path.join(self.log_dir, 'actions' + '.csv'), "w+")
        self.with_prediction_horizon = False
        if self.agent.prediction_horizon:
            self.with_prediction_horizon = True

    def run_episode(self, i):
        if self.with_prediction_horizon:
            episode = TestingEpisodePredictionHorizon(self.agent, self.env)
        else:
            episode = TestingEpisode(self.agent, self.env)
        super().run_episode(episode, i)
        for i in range (0, episode.stats['n_timesteps']):
            self.expected_values_file.write(str(episode.stats['q_value'][i]) + ',' + str(episode.stats['monte_carlo_return'][i]) + '\n')
            self.expected_values_file.flush()

        for i in range(len(episode.stats['actions_performed']) - 1):
            self.actions_file.write(str(episode.stats['actions_performed'][i]) + ',')
        self.actions_file.write(str(episode.stats['actions_performed'][-1]) + '\n')
        self.actions_file.flush()

    def run(self):
        super().run()

        # Write the evaluation results in file after evaluation
        self.episode_list_data.calc()
        with open(os.path.join(self.log_dir, 'eval_results.txt'), "w+") as file:
            file.write(self.episode_list_data.__str__())

class TrainEvalWorld(RLWorld):
    def __init__(self, agent, env, params, name=None):
        #n_episodes, render, save_every, eval_episodes, render_eval, eval_every
        super(TrainEvalWorld, self).__init__(agent, env, params, name)
        self.eval_episodes = params['eval_episodes']
        self.render_eval = params['eval_render']
        self.eval_every = params['eval_every']

        self.stats = Stats(self.sess, self.log_dir, self.tf_writer, 'train', self.agent.info)
        self.eval_stats = Stats(self.sess, self.log_dir, self.tf_writer, 'eval', self.agent.info)
        self.expected_values_file = open(os.path.join(self.log_dir, 'expected_values' + '.csv'), "w+")
        self.expected_values_file.write('expected,real\n')
        self.actions_file = open(os.path.join(self.log_dir, 'actions' + '.csv'), "w+")

        self.counter = 0

        self.episode_stats_eval = []

    def reset(self):
        super(TrainEvalWorld, self).reset()
        self.episode_stats_eval = []
        self.counter = 0

    def run_episode(self, i):
        episode = TrainingEpisode(self.agent, self.env)
        super(TrainEvalWorld, self).run_episode(episode, i)

        # Evaluate every some number of training episodes
        if (i + 1) % self.eval_every == 0:
            for j in range(self.eval_episodes):
                episode = TestingEpisode(self.agent, self.env)
                episode.run(self.render_eval)
                self.eval_stats.update(self.eval_episodes * self.counter + j, episode.stats)
                self.episode_stats_eval.append(episode.stats)
                pickle.dump(self.episode_stats_eval, open(os.path.join(self.log_dir, 'episode_stats_eval.pkl'), 'wb'))

                for k in range (0, episode.stats['n_timesteps']):
                    self.expected_values_file.write(str(episode.stats['q_value'][k]) + ',' + str(episode.stats['monte_carlo_return'][k]) + '\n')
                    self.expected_values_file.flush()

                for k in range(len(episode.stats['actions_performed']) - 1):
                    self.actions_file.write(str(episode.stats['actions_performed'][k]) + ',')
                self.actions_file.write(str(episode.stats['actions_performed'][-1]) + '\n')
                self.actions_file.flush()

            self.counter += 1

    def save(self, suffix=''):
        super(TrainEvalWorld, self).save(suffix)
        agent_model_file_name = os.path.join(self.log_dir, 'model' + suffix + '.pkl')
        self.agent.save(agent_model_file_name)

class SampleTransitionsWorld(RLWorld):
    def __init__(self, agent, env, params, name):
        super().__init__(agent, env, params, name)
        self.stats = Stats(self.sess, self.log_dir, self.tf_writer, 'train', self.agent.info)
        self.env_data = EnvData(['extra_data'])

        self.iteration_name = 'transitions'
        self.iterations = params[self.iteration_name]

    def reset(self):
        super().reset()
        self.env_data.reset()

    def run(self):
        logger.info('Sampling %d transitions from %s using %s.', self.iterations, self.env_name, self.agent_name)
        self.reset()
        self.set_state(WorldState.RUNNING)
        i = 0
        timesteps = 0
        while self.config['results']['progress'] < 1:

            episode = DataAcquisitionEpisode(self.agent, self.env)
            episode.run(self.render)
            timesteps += episode.stats['n_timesteps']

            # Update tensorboard stats
            self.stats.update(i, episode.stats)
            self.episode_stats.append(episode.stats)

            self.experience_time += episode.stats['experience_time']
            self.update_results(n_timesteps = episode.stats['n_timesteps'])

            self.env_data += episode.env_data
            self.env_data.save(self.log_dir)

            if self.stop_running:
                break

            i += 1

        while len(self.env_data.transitions) > self.iterations:
            del self.env_data.transitions[-1]
            del self.env_data.info['extra_data'][-1]
        self.env_data.save(self.log_dir)

        self.set_state(WorldState.FINISHED)

    def update_results(self, n_timesteps, thread_safe=True, write_yaml=True):

        # Update the results fields in config
        if thread_safe:
            self.results_lock.acquire()

        self.config['results']['n_timesteps'] += n_timesteps
        prog = self.config['results']['n_timesteps'] / self.iterations
        super(RLWorld, self).update_results(progress=prog, thread_safe=False, write_yaml=False)

        if thread_safe:
            self.results_lock.release()

        # Update YAML file
        if write_yaml:
            with open(os.path.join(self.log_dir, 'config.yml'), 'w') as outfile:
                yaml.dump(self.config, outfile, default_flow_style=False)

class SampleInitStatesWorld(RLWorld):
    def __init__(self, agent, env, params, name=None):
        super().__init__(agent, env, params, name)
        self.iteration_name = 'samples'
        self.iterations = params[self.iteration_name]
        self.env_data = EnvData()

    def reset(self):
        super().reset()
        self.env_data.reset()

    def run(self):
        logger.info('Sample %d init states from %s using %s.', self.iterations, self.agent_name, self.env_name)
        self.reset()
        self.set_state(WorldState.RUNNING)
        for i in range(self.iterations):
            self.env.reset()
            state = self.env.state_dict()
            self.env_data.init_states.append(state)
            self.env_data.save(self.log_dir)
            self.update_results(n_iterations = i + 1, n_timesteps = 0)
            if self.stop_running:
                break

        self.set_state(WorldState.FINISHED)

# Episode classes

class Episode:
    '''Represents a rollout of multiple timesteps of interaction between
    environment and agent'''
    def __init__(self, agent, env):
        self.env = env
        self.agent = agent

        # Set up stats dictionary
        self.stats = {}
        self.stats['n_timesteps'] = 0
        self.stats['experience_time'] = 0
        self.stats['reward'] = []
        self.stats['q_value'] = []
        self.stats['success'] = False
        self.stats['info'] = {}
        self.stats['actions_performed'] = []
        self.stats['monte_carlo_return'] = []
        for key in self.agent.info:
            self.stats['info'][key] = []

        self.data = EpisodeData()

    def run(self, render = False, init_state = None):

        self.env.load_state_dict(init_state)
        state = self.env.reset()

        while True:
            if (render):
                self.env.render()
            action = self._action_policy(state)
            next_state, reward, done, info = self.env.step(action)
            transition = Transition(state, action, reward, next_state, done)
            self._learn(transition)
            self._update_stats_step(transition, info)

            state = next_state.copy()
            if done:
                break

        self._update_states_episode(info)

    def _action_policy(self, state):
        raise NotImplementedError('Should be implemented by child classes.')

    def _learn(self, transition):
        raise NotImplementedError('Should be implemented by child classes.')

    def _update_stats_step(self, transition, info):
        self.stats['n_timesteps'] += 1
        if 'experience_time' in info:
            self.stats['experience_time'] += info['experience_time']
        self.stats['reward'].append(transition.reward)
        self.stats['q_value'].append(self.agent.q_value(transition.state, transition.action))
        for var in self.agent.info:
            self.stats['info'][var].append(self.agent.info[var])

        self.stats['actions_performed'].append(transition.action)
        self.stats['monte_carlo_return'].append(transition.reward)
        for i in range(0, len(self.stats['monte_carlo_return']) - 1):
            self.stats['monte_carlo_return'][i] += transition.reward

        timestep = TimestepData()
        timestep.transition = transition.copy()
        timestep.q_value = self.agent.q_value(transition.state, transition.action)
        self.data.append(timestep)

    def _update_states_episode(self, last_info):
        if 'success' in last_info:
            self.stats['success'] = last_info['success']
            self.data.success = last_info['success']

        for i in range (0, self.stats['n_timesteps']):
            self.stats['monte_carlo_return'][i] = self.stats['monte_carlo_return'][i] / (self.stats['n_timesteps'] - i)

class TrainingEpisode(Episode):
    def __init__(self, agent, env):
        super(TrainingEpisode, self).__init__(agent, env)

    def _action_policy(self, state):
        return self.agent.explore(state)

    def _learn(self, transition):
        self.agent.learn(transition)

class TestingEpisode(Episode):
    def _action_policy(self, state):
        return self.agent.predict(state)

    def _learn(self, transition):
        pass

class DataAcquisitionEpisode(Episode):
    def __init__(self, agent, env, buffer_size=1e6):
        super().__init__(agent, env)
        self.env_data = EnvData(['extra_data'])

    def _action_policy(self, state):
        return self.agent.predict(state)

    def _learn(self, transition):
        self.env_data.transitions.append(transition)

    def _update_stats_step(self, transition, info):
        super()._update_stats_step(transition, info)
        if 'extra_data' in info:
            self.env_data.info['extra_data'].append(info['extra_data'])

class TestingEpisodePredictionHorizon(Episode):
    """ Test episdoe with multiple augmented actions
    Differences with episode:
    - Assumes augmented action in transition
    """
    def run(self, render = False, init_state = None):

        self.env.load_state_dict(init_state)
        state = self.env.reset()

        while True:

            action = self.agent.predict(state)

            if self.agent.prediction_horizon == 1:
                action = [action]

            # Caution: This will not work with every gym env. reset_horizon()
            # exist only in specific envs like Clutter
            self.env.reset_horizon()

            for i in range(self.agent.prediction_horizon):

                if (render):
                    self.env.render()

                next_state, reward, done, info = self.env.step(action[i])

                # Assumes that the agent returns augmented action (dict)
                transition = Transition(state, action[i]['action'], reward, next_state, done)
                self._update_stats_step(transition, info)

                state = next_state.copy()
                if done:
                    break
            if done:
                break

        self._update_states_episode(info)
