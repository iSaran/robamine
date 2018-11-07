"""
Core
====

This module contains the core classes of the package for defining basic
interfaces. These base classes can be extended for implementing different RL
algorithms. Currently, the base classes for an RL agent are defined and for Neural Network in order to combine it with Deep Learning.
"""

import gym
import tensorflow as tf
from robamine.algo.util import DataStream, Stats, get_now_timestamp, print_progress
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
                 name=None):
        self.random_seed = random_seed
        self.name = name

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

    def __init__(self, sess, params=AgentParams()):
        self.params = params
        self.sess = sess

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def q_value(self, state, action):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def restore(self):
        raise NotImplementedError

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
            return self.sess.run(self.params.trainable)
        else:
            k = [v for v in self.params.trainable if v.name == name][0]
            return self.sess.run(k)

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
    def __init__(self, sess, agent, env, name=None):
        self.agent = agent
        self.env = env
        self.name = name

        self.state_dim = int(self.env.observation_space.shape[0])
        self.action_dim = int(self.env.action_space.shape[0])

        assert self.agent.params.state_dim == self.state_dim, 'Agent and environment has incompatible state dimension'
        assert self.agent.params.action_dim == self.action_dim, 'Agent and environment has incompantible action dimension'

        self.agent_name = self.agent.params.name
        self.env_name = self.env.spec.id

        self.log_dir = os.path.join(rb_logging.get_logger_path(), self.agent_name.replace(" ", "_") + '_' + self.env_name.replace(" ", "_"))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self._mode = WorldMode.EVAL

        tf_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        self.train_stats = Stats(sess, self.log_dir, tf_writer, 'train')
        self.eval_stats = Stats(sess, self.log_dir, tf_writer, 'eval')

        logger.info('Initialized agent: %s in environment: %s', self.agent.params.name, self.env.spec.id)

    @classmethod
    def create(cls, sess, agent_params, env_name, random_seed=999, name=None):
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
            agent = agent_handle(sess, agent_params, env.action_space)
        else:
            agent = agent_handle.create(sess, agent_params)

        return cls(sess, agent, env, name)

    def train(self, n_episodes, render=False):
        logger.info('%s training on %s for %d episodes', self.agent_name, self.env_name, n_episodes)
        self._mode = WorldMode.TRAIN
        self.run(n_episodes=n_episodes, episode_batch_size=10, render=render)

    def evaluate(self, n_episodes, render=False):
        logger.info('%s evaluating on %s for %d episodes', self.agent_name, self.env_name, n_episodes)
        self._mode = WorldMode.EVAL
        self.run(n_episodes=n_episodes, episode_batch_size=10, render=render)

    def train_and_eval(self, n_episodes_to_train, n_episodes_to_evaluate, evaluate_every, render_train=False, render_eval=False):
        logger.info('%s training on %s for %d episodes and evaluating for %d episodes every %d episodes of training', self.agent_name, self.env_name, n_episodes_to_train, n_episodes_to_evaluate, evaluate_every)
        self._mode = WorldMode.TRAIN_EVAL
        self.run(n_episodes=n_episodes_to_train, episode_batch_size=evaluate_every, episodes_to_evaluate=n_episodes_to_evaluate, render=render_train, render_eval=render_eval)

    def run(self, n_episodes, episode_batch_size=1, episodes_to_evaluate=0, render=False, render_eval=False):
        assert n_episodes % episode_batch_size == 0

        logger.debug('Start running world')
        if self._mode == WorldMode.TRAIN or self._mode == WorldMode.TRAIN_EVAL:
            train = True
        elif self._mode == WorldMode.EVAL:
            train = False

        counter = 0
        start_time = time.time()
        total_n_timesteps = 0
        for episode in range(n_episodes):

            logger.debug('Start episode: %d', episode)
            # Run an episode
            episode_stats, n_timesteps = self._episode(render, train)
            if train:
                self.train_stats.update(episode, episode_stats)
                total_n_timesteps += n_timesteps
            else:
                self.eval_stats.update(episode, episode_stats)

            # Store episode stats
            # if train:
            #     self.train_stats.update_episode(episode, episode_stats)
            # else:
            #     self.eval_stats.update_episode(episode, episode_stats)

            if (episode + 1) % episode_batch_size == 0:
                if self._mode == WorldMode.TRAIN_EVAL:
                    for eval_episode in range(episodes_to_evaluate):
                        episode_stats, _ = self._episode(render_eval, train=False)
                        self.eval_stats.update(episodes_to_evaluate * counter + eval_episode, episode_stats)
                counter += 1
                        # self.eval_stats.update_episode()
                print_progress(episode, n_episodes, start_time, total_n_timesteps, 0.02)
                # self.logger.flush()

    def _episode(self, render=False, train=False):
        stats = {'reward': [], 'q_value': []}
        state = self.env.reset()
        while True:
            if (render):
                self.env.render()

            logger.debug('Timestep %d: ', len(stats['reward']))
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
            stats['reward'].append(reward)
            stats['q_value'].append(Q)

            if done:
                break

        return stats, len(stats['reward'])

