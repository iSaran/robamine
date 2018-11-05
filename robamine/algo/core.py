"""
Core
====

This module contains the core classes of the package for defining basic
interfaces. These base classes can be extended for implementing different RL
algorithms. Currently, the base classes for an RL agent are defined and for Neural Network in order to combine it with Deep Learning.
"""

import gym
import tensorflow as tf
from robamine.algo.util import Logger, Stats, get_now_timestamp
from robamine import rb_logging
import logging
import os
import pickle

logger = logging.getLogger('robamine.algo.core')

class AgentParams():
    def __init__(self):
        self.env_name = None
        self.state_dim = None
        self.action_dim = None
        self.episode_horizon = None
        self.name = None

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

    def __init__(self, sess, env, random_seed=999, log_dir='/tmp', params=None):
        if params:
            self.params = params
        else:
            self.params = AgentParams()

        # Environment setup
        self.params.env_name = env
        self.env = gym.make(self.params.env_name)
        self.env.seed(random_seed)
        if isinstance(self.env.observation_space, gym.spaces.dict_space.Dict):
            logger.warn('Gym environment has a %s observation space. I will wrap it with a gym.wrappers.FlattenDictWrapper.', type(self.env.observation_space))
            self.env = gym.wrappers.FlattenDictWrapper(self.env, ['observation', 'desired_goal'])

        self.params.state_dim = int(self.env.observation_space.shape[0])
        self.params.action_dim = int(self.env.action_space.shape[0])
        self.params.episode_horizon = int(self.env._max_episode_steps)

        self.sess = sess

        self.log_dir = os.path.join(rb_logging.get_logger_path(), self.name.replace(" ", "_") + '_' + self.env.spec.id.replace(" ", "_"))

        logger.info('Initialized agent: %s in environment: %s', self.name, self.env.spec.id)

    def train(self, n_episodes, episode_batch_size = 1, render=False, episodes_to_evaluate=0, render_eval = False):
        """
        Performs the policy improvement loop. For a given number of episodes
        this function runs the basic RL loop for each timestep of the episode,
        i.e.:

        * Selects an action based on the exploration policy (see :meth:`.explore`)
        * Performs the action to the environment and reads the next state
          and the reward.
        * Learns from this experience (i.e. optimizes the learned policy
          based on some algorithm, see :meth:`.learn`).

        Parameters
        ----------
        n_episodes : int
            The number of episodes to train the model.
        episode_batch_size : int
            The number of episodes per episode batch. Used to divide episode
            for extracting valuable statistics
        render : bool
            True of rendering is required. False otherwise.
        episodes_to_evaluate : int
            The number of episodes to evaluate the agent during training (at the end of each epoch)
        """

        assert n_episodes % episode_batch_size == 0

        logger.debug('Agent: starting training')
        for episode in range(n_episodes):
            state = self.env.reset()
            logger.debug('Episode: %d',  episode)

            for t in range(self.episode_horizon):

                if (render):
                    self.env.render()

                logger.debug('Timestep %d: ', t)
                logger.debug('Given state: %s', state.__str__())

                #self.logger.console.debug('Actor params:' + self.actor.get_params().__str__())
                action = self.explore(state)
                logger.debug('Action to explore: %s', action.__str__())

                # Execute the action on the environment and observe reward and next state
                next_state, reward, done, info = self.env.step(action)
                logger.debug('Next state by the environment: %s', next_state.__str__())
                logger.debug('Reward by the environment: %f', reward)

                # Learn
                logger.debug('Learn based on this transition')
                self.learn(state, action, reward, next_state, done)
                Q = self.q_value(state, action)
                logger.debug('Q value by the agent: %f', Q)

                state = next_state

                self.train_stats.update_timestep({'reward': reward, 'q_value': Q})

                if done:
                    break

            self.train_stats.update_episode(episode)

            if ((episode + 1) % episode_batch_size == 0):
                self.train_stats.update_batch((episode + 1) / episode_batch_size - 1)
                self.train_stats.print_progress(self.name, self.env.spec.id, episode, n_episodes)
                self.evaluate(episodes_to_evaluate, render_eval)
                self.logger.flush()
                # self.save()

            # Evaluate the agent (run the learned policy for a number of episodes)
            #eval_stats = self.evaluate(episodes_to_evaluate)

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

    def evaluate(self, n_episodes = 1, render=False):
        """
        Performs the policy evaluation loop. For a given number of episodes
        this function runs the basic RL loop for each timestep of the episode
        performing only optimal actions (no exploration as it happens in
        :meth:`.train`):

        * Selects an action based on the optimal policy (calls :meth:`.predict`)
        * Performs the action to the environment and reads the next state
          and the reward.

        Parameters
        ----------
        n_episodes : int
            The number of episodes to evaluate the model. Multiple episodes can be used for producing stats, average of successes etc.
        render : bool
            True of rendering is required. False otherwise.
        """
        if (n_episodes == 0):
            return

        logger.debug('Agent: Starting evalulation for %d episodes.', n_episodes)

        for episode in range(n_episodes):
            state = self.env.reset()

            logger.debug('Episode: %d', episode)

            for t in range(self.episode_horizon):

                logger.debug('Timestep: %d', t)

                if (render):
                    self.env.render()

                # Select an action based on the exploration policy
                action = self.predict(state)

                # Execute the action on the environment  and observe reward and next state
                next_state, reward, done, info = self.env.step(action)

                Q = self.q_value(state, action)

                state = next_state

                self.eval_stats.update_timestep({'reward': reward, 'q_value': Q})

                if done:
                    break

            self.eval_stats.update_episode(n_episodes * self.eval_episode_batch + episode)

        self.eval_stats.update_batch(self.eval_episode_batch)
        self.eval_episode_batch += 1

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

class NetworkParams(input_dim=None,
                    out_dim=None,
                    hidden_units=None,
                    trainable=None,
                    name=None):
    def __init__(self):
        self.input_dim = input_dim
        self.output_dim = out_dim
        self.hidden_units = hidden_units
        self.trainable = trainable
        self.name = name

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

    def __init__(self, sess, params):
        self.sess = sess

        if params:
            self.params = params
        else:
            self.params = NetworkParams()

        self.name = self.params.name

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
            return self.sess.run(self.net_params)
        else:
            k = [v for v in self.net_params if v.name == name][0]
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
