"""
Core
====

This module contains the core classes of the package for defining basic
interfaces. These base classes can be extended for implementing different RL
algorithms. Currently, the base classes for an RL agent are defined and for Neural Network in order to combine it with Deep Learning.
"""

import gym
import tensorflow as tf
from robamine.algo.util import DataStream, Stats, get_now_timestamp, print_progress, EpisodeStats, Plotter
from robamine import rb_logging
import logging
import os
import pickle
from enum import Enum
import importlib
import time

logger = logging.getLogger('robamine.algo.core')

class AgentParams:
    def __init__(self,
                 random_seed=999,
                 name=None,
                 suffix=""):
        self.random_seed = random_seed
        self.name = name
        self.suffix=suffix

        self.state_dim=None,
        self.action_dim=None,

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

    def __init__(self, params=AgentParams()):
        self.params = params
        self.sess = tf.Session()

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

    def to_dict(self):
        return {'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'out_dim': self.out_dim,
                'name': self.name,
                'inputs': self.inputs,
                'out': self.out,
                'net_params': self.net_params
               }

    @classmethod
    def from_dict(cls, sess, data):
        net = cls(sess, data['input_dim'], data['hidden_dims'], data['out_dim'], data['name'], \
                        data['inputs'], data['out'], data['net_params'])
        return cls

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

class WorldMode(Enum):
    TRAIN = 1
    EVAL = 2
    TRAIN_EVAL = 3

class World:
    def __init__(self, agent, env, name=None):
        try:
            assert isinstance(agent, Agent), 'World: The given agent is not an Agent object. Use World.create() instead.'
            assert isinstance(env, gym.Env), 'World: The given environment is not a Gym Env object. Use World.create() instead.'
        except AssertionError as err:
            logger.exception(err)
            raise err

        self.agent = agent
        self.env = env
        self.name = name

        self.state_dim = int(self.env.observation_space.shape[0])
        self.action_dim = int(self.env.action_space.shape[0])

        assert self.agent.params.state_dim == self.state_dim, 'Agent and environment has incompatible state dimension'
        assert self.agent.params.action_dim == self.action_dim, 'Agent and environment has incompantible action dimension'

        self.agent_name = self.agent.params.name + self.agent.params.suffix
        self.env_name = self.env.spec.id

        self.log_dir = os.path.join(rb_logging.get_logger_path(), self.agent_name.replace(" ", "_") + '_' + self.env_name.replace(" ", "_"))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self._mode = WorldMode.EVAL

        self.tf_writer = tf.summary.FileWriter(self.log_dir, agent.sess.graph)
        self.train_stats = None
        self.eval_stats = None

        logger.info('Initialized agent: %s in environment: %s', self.agent_name, self.env.spec.id)

    @classmethod
    def create(cls, agent_params, env_name, random_seed=999, name=None):
        # Environment setup
        env = gym.make(env_name)
        env.seed(random_seed)
        if isinstance(env.observation_space, gym.spaces.dict_space.Dict):
            logger.warn('Gym environment has a %s observation space. I will wrap it with a gym.wrappers.FlattenDictWrapper.', type(env.observation_space))
            env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'desired_goal'])

        # Setup agent
        agent_params.state_dim = int(env.observation_space.shape[0])
        agent_params.action_dim = int(env.action_space.shape[0])
        agent_params.random_seed = random_seed

        module = importlib.import_module('robamine.algo.' + agent_params.name.lower())
        agent_handle = getattr(module, agent_params.name)

        if (agent_params.name == 'Dummy'):
            agent = agent_handle(agent_params, env.action_space)
        else:
            agent = agent_handle.create(agent_params)

        return cls(agent, env, name)

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
            self.train_stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'train')
        elif self._mode == WorldMode.EVAL:
            train = False
            self.eval_stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'eval')
        elif self._mode == WorldMode.TRAIN_EVAL:
            train = True
            self.train_stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'train')
            self.eval_stats = Stats(self.agent.sess, self.log_dir, self.tf_writer, 'eval')

        counter = 0
        start_time = time.time()
        total_n_timesteps = 0
        for episode in range(n_episodes):

            logger.debug('Start episode: %d', episode)
            # Run an episode
            episode_stats = self._episode(render, train)
            if train:
                self.train_stats.update(episode, episode_stats)
                total_n_timesteps += episode_stats.n_timesteps
            else:
                self.eval_stats.update(episode, episode_stats)

            # Store episode stats
            # if train:
            #     self.train_stats.update_episode(episode, episode_stats)
            # else:
            #     self.eval_stats.update_episode(episode, episode_stats)

            # Evaluate every some number of training episodes
            if evaluate_every and (episode + 1) % evaluate_every == 0 and episodes_to_evaluate:
                for eval_episode in range(episodes_to_evaluate):
                    episode_stats = self._episode(render_eval, train=False)
                    self.eval_stats.update(episodes_to_evaluate * counter + eval_episode, episode_stats)
                counter += 1

            # Print progress every print_progress_every episodes
            if print_progress_every and (episode + 1) % print_progress_every == 0:
                print_progress(episode, n_episodes, start_time, total_n_timesteps, 0.02)

            if save_every and (episode + 1) % save_every == 0:
                self.agent.save(os.path.join(self.log_dir, 'model.pkl'))


    def _episode(self, render=False, train=False):
        stats = EpisodeStats()
        state = self.env.reset()
        while True:
            if (render):
                self.env.render()

            logger.debug('Timestep %d: ', len(stats.reward))
            logger.debug('Given state: %s', state.__str__())

            # Act: Explore or optimal policy?
            if train:
                action = self.agent.explore(state)
            else:
                action = self.agent.predict(state)

            logger.debug('Action to explore: %s', action.__str__())

            # Execute the action on the environment and observe reward and next state
            next_state, reward, done, info = self.env.step(action)
            logger.debug('Next state by the environment: %s', next_state.__str__())
            logger.debug('Reward by the environment: %f', reward)

            transition = Transition(state, action, reward, next_state, done)

            if train:
                self.agent.learn(transition)

            # Learn
            logger.debug('Learn based on this transition')
            Q = self.agent.q_value(state, action)
            logger.debug('Q value by the agent: %f', Q)

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
