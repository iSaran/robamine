"""
Core
====

This module contains the core classes of the package for defining basic
interfaces. These base classes can be extended for implementing different RL
algorithms. Currently, the base classes for an RL agent are defined and for Neural Network in order to combine it with Deep Learning.
"""

import gym
import tensorflow as tf
from robamine.algo.util import DataStream, Stats, get_now_timestamp, print_progress, EpisodeStats, Plotter, get_agent_handle
from robamine import rb_logging
import logging
import os
import pickle
from enum import Enum
import time
import numpy as np

logger = logging.getLogger('robamine.algo.core')

class AgentParams:
    def __init__(self,
                 state_dim=None,
                 action_dim=None,
                 name=None,
                 suffix=""):
        self.name = name
        self.suffix = suffix

        # Store internally the state_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

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

    def __init__(self, state_dim, action_dim, params=AgentParams()):
        self.state_dim, self.action_dim  = state_dim, action_dim
        self.params = params
        self.params.state_dim, self.params.action_dim  = state_dim, action_dim
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
        error = 'Agent ' + self.params.name + ' does not implement an exploration policy.'
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
        error = 'Agent ' + self.params.name + ' does not implemement a learning algorithm.'
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
        error = 'Agent ' + self.params.name + ' does not implement a policy.'
        logger.error(error)
        raise NotImplementedError(error)

    def q_value(self, state, action):
        error = 'Agent ' + self.params.name + ' does not provide a Q value.'
        logger.error(error)
        raise NotImplementedError(error)

    def save(self, file_path):
        error = 'Agent ' + self.params.name + ' does not provide saving capabilities.'
        logger.error(error)
        raise NotImplementedError(error)

    def load(self):
        error = 'Agent ' + self.params.name + ' does not provide loading capabilities.'
        logger.error(error)
        raise NotImplementedError(error)

    def seed(self, seed):
        error = 'Agent ' + self.params.name + ' does cannot be seeded.'
        logger.error(error)
        raise NotImplementedError(error)

class NetworkParams:
    def __init__(self,
                 input_dim=None,
                 hidden_units=None,
                 output_dim=None,
                 name=None):
        self.hidden_units = hidden_units
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.trainable = None

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

    def __init__(self, sess, params = NetworkParams()):
        self.sess = sess
        self.params = params

        self.input = None
        self.out = None
        self.net_params = None

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
    def load(cls, sess, params):
        self = cls.create(sess, params)
        sess.run([self.net_params[i].assign(self.params.trainable[i]) for i in range(len(self.net_params))])
        return self

class Transition:
    def __init__(self,
                 state=None,
                 action=None,
                 reward=None,
                 next_state=None,
                 terminal=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal

    def array(self):
        return np.array([self.state, self.action, self.reward, self.next_state, self.terminal])

    def __str__(self):
        return '[state: ' + str(self.state) + \
                ', action: ' + str(self.action) + \
                ', reward: ' + str(self.reward) + \
                ', next_state: ' + str(self.next_state) + \
                ', terminal: ' + str(self.terminal) + ']'

class WorldMode(Enum):
    TRAIN = 1
    EVAL = 2
    TRAIN_EVAL = 3

class World:
    def __init__(self, agent, env, name=None):
        # Environment setup
        if isinstance(env, gym.Env):
            self.env = env
        elif isinstance(env, str):
            self.env = gym.make(env)
            if isinstance(self.env.observation_space, gym.spaces.dict.Dict):
                logger.warn('Gym environment has a %s observation space. I will wrap it with a gym.wrappers.FlattenDictWrapper.', type(self.env.observation_space))
                self.env = gym.wrappers.FlattenDictWrapper(self.env, ['observation', 'desired_goal'])
        else:
            err = ValueError('Provide a gym.Env or a string in order to create a new world')
            logger.exception(err)
            raise err

        self.state_dim = int(self.env.observation_space.shape[0])
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = int(self.env.action_space.shape[0])

        # Agent setup
        if isinstance(agent, str):
            agent_name = agent
            agent_handle, agent_params_handle = get_agent_handle(agent_name)
            agent_params = agent_params_handle()
            if (agent_params.name == 'Dummy'):
                self.agent = agent_handle(self.env.action_space, self.state_dim, self.action_dim, agent_params)
            else:
                self.agent = agent_handle(self.state_dim, self.action_dim, agent_params)
        elif isinstance(agent, AgentParams):
            agent_params = agent
            agent_handle, _ = get_agent_handle(agent_params.name)
            if (agent_params.name == 'Dummy'):
                self.agent = agent_handle(self.env.action_space, self.state_dim, self.action_dim, agent_params)
            else:
                self.agent = agent_handle(self.state_dim, self.action_dim, agent_params)
        elif isinstance(agent, Agent):
            self.agent = agent
        else:
            err = ValueError('Provide a Agent, AgentParams, or a string in order to create an agent for the new world')
            logger.exception(err)
            raise err

        self.name = name

        try:
            assert self.agent.params.state_dim == self.state_dim, 'Agent and environment has incompatible state dimension'
            assert self.agent.params.action_dim == self.action_dim, 'Agent and environment has incompantible action dimension'
        except AssertionError as err:
            logger.exception(err)
            raise err

        self.agent_name = self.agent.params.name + self.agent.params.suffix
        self.env_name = self.env.spec.id

        self.log_dir = os.path.join(rb_logging.get_logger_path(), self.agent_name.replace(" ", "_") + '_' + self.env_name.replace(" ", "_"))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self._mode = WorldMode.EVAL

        self.tf_writer = tf.summary.FileWriter(self.log_dir, self.agent.sess.graph)
        self.train_stats = None
        self.eval_stats = None

        logger.info('Initialized world with the %s in the %s environment', self.agent_name, self.env.spec.id)

    def seed(self, seed):
        self.env.seed(seed)
        self.agent.seed(seed)

    def train(self, n_episodes, render=False, print_progress_every=1, save_every=None):
        logger.info('%s training on %s for %d episodes', self.agent_name, self.env_name, n_episodes)
        self._mode = WorldMode.TRAIN
        self._run(n_episodes, None, None, print_progress_every, render, False, save_every)

    def evaluate(self, n_episodes, render=False, print_progress_every=1):
        logger.info('%s evaluating on %s for %d episodes', self.agent_name, self.env_name, n_episodes)
        self._mode = WorldMode.EVAL
        self._run(n_episodes, None, None, print_progress_every, render, False, None)

    def train_and_eval(self, n_episodes_to_train, n_episodes_to_evaluate, evaluate_every, render_train=False, render_eval=False, print_progress_every=1, save_every=None):
        logger.info('%s training on %s for %d episodes and evaluating for %d episodes every %d episodes of training', self.agent_name, self.env_name, n_episodes_to_train, n_episodes_to_evaluate, evaluate_every)
        self._mode = WorldMode.TRAIN_EVAL
        self._run(n_episodes_to_train, evaluate_every, n_episodes_to_evaluate, print_progress_every, render_train, render_eval, save_every)

    def _run(self, n_episodes, evaluate_every, episodes_to_evaluate, print_progress_every, render, render_eval, save_every):
        if evaluate_every:
            assert n_episodes % evaluate_every == 0

        logger.debug('Start running world')
        if self._mode == WorldMode.TRAIN:
            train = True
            self.train_stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'train', self.agent.info)
        elif self._mode == WorldMode.EVAL:
            train = False
            self.eval_stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'eval', self.agent.info)
        elif self._mode == WorldMode.TRAIN_EVAL:
            train = True
            self.train_stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'train')
            self.eval_stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'eval')

        counter = 0
        start_time = time.time()
        total_n_timesteps = 0
        experience_time = 0.0
        for episode in range(n_episodes):

            logger.debug('Start episode: %d', episode)
            # Run an episode
            episode_stats = self._episode(render, train)
            experience_time += episode_stats.experience_time
            if train:
                self.train_stats.update(episode, episode_stats)
                total_n_timesteps += episode_stats.n_timesteps
            else:
                self.eval_stats.update(episode, episode_stats)

            # Evaluate every some number of training episodes
            if evaluate_every and (episode + 1) % evaluate_every == 0 and episodes_to_evaluate:
                for eval_episode in range(episodes_to_evaluate):
                    episode_stats = self._episode(render_eval, train=False)
                    self.eval_stats.update(episodes_to_evaluate * counter + eval_episode, episode_stats)
                counter += 1

            # Print progress every print_progress_every episodes
            if print_progress_every and (episode + 1) % print_progress_every == 0:
                print_progress(episode, n_episodes, start_time, total_n_timesteps, experience_time)

            if save_every and (episode + 1) % save_every == 0:
                self.save()


    def _episode(self, render=False, train=False):
        stats = EpisodeStats(self.agent.info)
        state = self.env.reset()
        while True:
            if (render):
                self.env.render()

            # Act: Explore or optimal policy?
            if train:
                action = self.agent.explore(state)
            else:
                action = self.agent.predict(state)

            # Execute the action on the environment and observe reward and next state
            next_state, reward, done, info = self.env.step(action)
            if 'experience_time' in info:
                stats.experience_time += info['experience_time']

            if train:
                transition = Transition(state, action, reward, next_state, done)
                self.agent.learn(transition)

            for var in self.agent.info:
                stats.info[var].append(self.agent.info[var])

            # Learn
            Q = self.agent.q_value(state, action)

            state = next_state

            # self.train_stats.update_timestep({'reward': reward, 'q_value': Q})

            # Compile stats
            stats.reward.append(reward)
            stats.q_value.append(Q)

            if done:
                break

        stats.n_timesteps = len(stats.reward)
        return stats

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
        world = pickle.load(open(os.path.join(directory, 'world.pkl'), 'rb'))
        agent_name = world['agent_name']
        agent_handle, agent_params_handle = get_agent_handle(agent_name)
        agent_params = agent_params_handle()
        agent = agent_handle.load(os.path.join(directory, 'model.pkl'))

        self = cls(agent, world['env'])
        logger.info('World loaded from %s', directory)
        return self

    def save(self):
        self.agent.save(os.path.join(self.log_dir, 'model.pkl'))
        world_path = os.path.join(self.log_dir, 'world.pkl')
        world = {}
        world['env'] = self.env
        world['agent_name'] = self.agent_name
        pickle.dump(world, open(world_path, 'wb'))
        logger.info('World saved to %s', world_path)

