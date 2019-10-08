"""
Core
====

This module contains the core classes of the package for defining basic
interfaces. These base classes can be extended for implementing different RL
algorithms. Currently, the base classes for an RL agent are defined and for Neural Network in order to combine it with Deep Learning.
"""

import gym
import tensorflow as tf
from robamine.algo.util import DataStream, Stats, get_now_timestamp, print_progress, Plotter, get_agent_handle, transform_sec_to_timestamp, Transition
from robamine.utils.info import get_pc_and_version, get_dir_size
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

logger = logging.getLogger('robamine.algo.core')

class Agent:
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
        self.state_dim, self.action_dim  = state_dim, action_dim
        self.name = name
        self.params = params.copy()
        self.sess = tf.Session()
        self.info = {}

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

    def q_value(self, state, action):
        error = 'Agent ' + self.name + ' does not provide a Q value.'
        logger.error(error)
        raise NotImplementedError(error)

    def save(self, file_path):
        error = 'Agent ' + self.name + ' does not provide saving capabilities.'
        logger.error(error)
        raise NotImplementedError(error)

    def load(self):
        error = 'Agent ' + self.name + ' does not provide loading capabilities.'
        logger.error(error)
        raise NotImplementedError(error)

    def load_trainable(self):
        error = 'Agent ' + self.name + ' does not provide loading capabilities.'
        logger.error(error)
        raise NotImplementedError(error)

    def seed(self, seed):
        error = 'Agent ' + self.name + ' does cannot be seeded.'
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

# World classes

class World:
    def __init__(self, agent, env, n_episodes, render, save_every, print_every, name=None):
        # Environment setup
        if isinstance(env, gym.Env):
            self.env = env
        elif isinstance(env, str):
            self.env = gym.make(env)
            if isinstance(self.env.observation_space, gym.spaces.dict.Dict):
                logger.warn('Gym environment has a %s observation space. I will wrap it with a gym.wrappers.FlattenDictWrapper.', type(self.env.observation_space))
                self.env = gym.wrappers.FlattenDictWrapper(self.env, ['observation', 'desired_goal'])
        elif isinstance(env, dict):
            self.env = gym.make(env['name'], params=env)
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
        elif isinstance(agent, dict):
            agent_name = agent['name']
            agent_handle = get_agent_handle(agent_name)
            agent_params = agent['params'] if 'params' in agent else {}
            self.agent = agent_handle(self.state_dim, self.action_dim, agent_params)
        elif isinstance(agent, Agent):
            self.agent = agent
        else:
            err = ValueError('Provide an Agent or a string in order to create an agent for the new world')
            logger.exception(err)
            raise err
        self.agent_name = self.agent.name

        self.name = name

        # Check if environment and agent are compatible
        try:
            assert self.agent.state_dim == self.state_dim, 'Agent and environment has incompatible state dimension'
            assert self.agent.action_dim == self.action_dim, 'Agent and environment has incompantible action dimension'
        except AssertionError as err:
            logger.exception(err)
            raise err

        # Setup logging directory and tf writers
        self.log_dir = rb_logging.get_logger_path()
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.tf_writer = tf.summary.FileWriter(self.log_dir, self.agent.sess.graph)
        self.stats = None

        # Setup the internal config dictionary
        self.config = {}
        self.config['results'] = {}
        self.config['results']['logging_dir'] = self.log_dir
        self.config['results']['n_episodes'] = 0
        self.config['results']['n_timesteps'] = 0
        hostname, username, version = get_pc_and_version()
        self.config['results']['hostname'] = hostname + ':' + username
        self.config['results']['version'] = version
        self.config['results']['started_on'] = None
        self.config['results']['estimated_time'] = None
        self.config['results']['time_elapsed'] = None
        self.config['results']['dir_size'] = 0

        logger.info('Initialized world with the %s in the %s environment', self.agent_name, self.env.spec.id)

        self.episodes = n_episodes
        self.render = render
        self.save_every = save_every
        self.print_every = print_every

        # Other variables for running episodes
        self.experience_time = 0.0
        self.start_time = None

        # A list in which dictionaries for episodes stats are stored
        self.episode_stats = []

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
        self = cls(config['agent'], env, config['world']['episodes'], config['world']['render'], config['world']['save_every'], 1)

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

    def init(self):
        self.experience_time = 0
        self.start_time = time.time()
        self.config['results']['started_on'] = str(datetime.datetime.now())
        self.episode_stats = []

    def run_episode(self, episode, i):
        # Run the episode. Assumed that the episode has been already created by
        # child classes
        episode.run(self.render)

        # Update tensorboard stats
        self.stats.update(i, episode.stats)
        self.episode_stats.append(episode.stats)

        # Save agent model
        self.save()
        if self.save_every and (i + 1) % self.save_every == 0:
            self.save(suffix='_' + str(i+1))

        # Save the config in YAML file
        self.experience_time += episode.stats['experience_time']
        self.config['results']['n_episodes'] = i + 1
        self.config['results']['n_timesteps'] += episode.stats['n_timesteps']
        self.config['results']['time_elapsed'] = transform_sec_to_timestamp(time.time() - self.start_time)
        self.config['results']['experience_time'] = transform_sec_to_timestamp(self.experience_time)
        self.config['results']['estimated_time'] = transform_sec_to_timestamp((self.episodes - i + 1) * (time.time() - self.start_time) / (i + 1))
        self.config['results']['dir_size'] = get_dir_size(self.log_dir)
        with open(os.path.join(self.log_dir, 'config.yml'), 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

    def run(self):
        logger.info('%s running on %s for %d episodes', self.agent_name, self.env_name, self.episodes)
        self.init()
        for i in range(self.episodes):
            self.run_episode(i)

            # Print progress
            if self.print_every and (i + 1) % self.print_every == 0:
                print_progress(i, self.episodes, self.start_time, self.config['results']['n_timesteps'], self.experience_time)

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

class TrainWorld(World):
    def __init__(self, agent, env, n_episodes, render, save_every, print_every, name=None):
        super(TrainWorld, self).__init__(agent, env, n_episodes, render, save_every, print_every, name)
        self.stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'train', self.agent.info)

    def run_episode(self, i):
        episode = TrainingEpisode(self.agent, self.env)
        super(TrainWorld, self).run_episode(episode, i)

    def save(self, suffix=''):
        super(TrainWorld, self).save(suffix)
        agent_model_file_name = os.path.join(self.log_dir, 'model' + suffix + '.pkl')
        self.agent.save(agent_model_file_name)

class EvalWorld(World):
    def __init__(self, agent, env, n_episodes, render, save_every, print_every, name=None):
        super(EvalWorld, self).__init__(agent, env, n_episodes, render, save_every, print_every, name)
        self.stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'eval', self.agent.info)
        self.expected_values_file = open(os.path.join(self.log_dir, 'expected_values' + '.csv'), "w+")
        self.expected_values_file.write('expected,real\n')
        self.actions_file = open(os.path.join(self.log_dir, 'actions' + '.csv'), "w+")

    def run_episode(self, i):
        episode = TestingEpisode(self.agent, self.env)
        super(EvalWorld, self).run_episode(episode, i)
        for i in range (0, episode.stats['n_timesteps']):
            self.expected_values_file.write(str(episode.stats['q_value'][i]) + ',' + str(episode.stats['monte_carlo_return'][i]) + '\n')
            self.expected_values_file.flush()

        for i in range(len(episode.stats['actions_performed']) - 1):
            self.actions_file.write(str(episode.stats['actions_performed'][i]) + ',')
        self.actions_file.write(str(episode.stats['actions_performed'][-1]) + '\n')
        self.actions_file.flush()

class TrainEvalWorld(World):
    def __init__(self, agent, env, n_episodes, render, save_every, print_every, eval_episodes, render_eval, eval_every, name=None):
        super(TrainEvalWorld, self).__init__(agent, env, n_episodes, render, save_every, print_every, name)
        self.eval_episodes = eval_episodes
        self.render_eval = render_eval
        self.eval_every = eval_every

        self.stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'train', self.agent.info)
        self.eval_stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'eval', self.agent.info)
        self.expected_values_file = open(os.path.join(self.log_dir, 'expected_values' + '.csv'), "w+")
        self.expected_values_file.write('expected,real\n')
        self.actions_file = open(os.path.join(self.log_dir, 'actions' + '.csv'), "w+")

        self.counter = 0

        self.episode_stats_eval = []

    def init(self):
        super(TrainEvalWorld, self).init()
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
        self = cls(config['agent'], env, config['world']['episodes'], config['world']['render'], config['world']['save_every'], 0, config['world']['eval']['episodes'], config['world']['eval']['render'], config['world']['eval']['every'])

        if 'trainable_params' in config['agent'] and config['agent']['trainable_params'] != '':
            self.agent.load_trainable(config['agent']['trainable_params'])

        # Save the config
        self.config['world'] = config['world'].copy()
        self.config['env'] = config['env'].copy()
        self.config['agent'] = config['agent'].copy()

        return self

class DataAcquisitionWorld(World):
    def __init__(self, agent, env, n_episodes, render, save_every, print_every):
        super(DataAcquisitionWorld, self).__init__(agent, env, n_episodes, render, save_every, print_every)
        self.stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'train', self.agent.info)
        self.data = ReplayBuffer(1e6)

    def run_episode(self, i):
        episode = DataAcquisitionEpisode(self.agent, self.env)
        super(DataAcquisitionWorld, self).run_episode(episode, i)
        self.data.merge(episode.buffer)
        self.data.save(os.path.join(self.log_dir, 'data.pkl'))

    def save(self, suffix=''):
        pass

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

    def run(self, render):
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

    def _update_states_episode(self, last_info):
        if 'success' in last_info:
            self.stats['success'] = last_info['success']

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
    def __init__(self, agent, env):
        super(TestingEpisode, self).__init__(agent, env)

    def _action_policy(self, state):
        return self.agent.predict(state)

    def _learn(self, transition):
        pass

class DataAcquisitionEpisode(Episode):
    def __init__(self, agent, env, buffer_size=1e6):
        super(DataAcquisitionEpisode, self).__init__(agent, env)
        self.buffer = ReplayBuffer(buffer_size)

    def _action_policy(self, state):
        return self.agent.predict(state)

    def _learn(self, transition):
        self.buffer.store(transition)
