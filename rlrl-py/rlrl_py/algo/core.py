"""
Core
====

This module contains the core classes of the package for defining basic
interfaces. These base classes can be extended for implementing different RL
algorithms. Currently, the base classes for an RL agent are defined and for Neural Network in order to combine it with Deep Learning.
"""

import gym
import tensorflow as tf
import rlrl_py.algo.util as util

class Agent:
    """
    Base class for creating an RL agent.

    Example
    -------
    Assume you want to implement the Best RL Algorithm (BRLA). Create a class which inherits from :class:`.Agent` and implement its member methods. Then, you should be able to train this agent for 1 million episodes in the ``MyRobot`` environment as simple as:

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

    def __init__(self, sess, env, random_seed=1e6, log_dir='/tmp', name=None):
        # Environment setup
        self.env_name = env
        self.env = gym.make(env)
        # TODO(isaran): Maybe a wrapper needs for the goal environments
        # self.env = gym.wrappers.FlattenDictWrapper(self.env, ['observation', 'desired_goal'])
        self.env.seed(random_seed)
        self.episode_horizon = int(self.env._max_episode_steps)

        self.sess = sess
        self.name = name

        self.logger = util.Logger(sess, log_dir, self.name, env)

    def train(self, n_episodes, n_epochs = 1, render=False, episodes_to_evaluate=0):
        """
        Performs the policy improvement loop. For a given number of episodes this function runs the basic RL loop for each timestep of the episode, i.e.:

        * Selects an action based on the exploration policy (see :meth:`.explore`)
        * Performs the action to the environment and reads the next state
          and the reward.
        * Learns from this experience (i.e. optimizes the learned policy
          based on some algorithm, see :meth:`.learn`).

        Parameters
        ----------
        n_episodes : int
            The number of episodes to train the model.
        n_epochs : int
            The number of epochs to divide the episodes.
        render : bool
            True of rendering is required. False otherwise.
        episodes_to_evaluate : int
            The number of episodes to evaluate the agent during training (at the end of each epoch)
        """

        assert n_episodes % n_epochs == 0
        n_episodes_per_epoch = int(n_episodes / n_epochs)

        stats = util.Stats(self.name, self.env_name, n_episodes, n_epochs, dt=0.02, logger=self.logger, name = "Training")

        for epoch in range(n_epochs):
            stats.init_for_epoch()
            for episode in range(n_episodes_per_epoch):
                stats.init_for_episode()
                state = self.env.reset()

                for t in range(self.episode_horizon):

                    stats.init_for_timestep()

                    if (render):
                        self.env.render()

                    # Select an action based on the exploration policy
                    action = self.explore(state)

                    # Execute the action on the environment  and observe reward and next state
                    next_state, reward, done, info = self.env.step(action)

                    # Learn
                    Q = self.learn(state, action, reward, next_state, done)

                    state = next_state

                    stats.update_for_timestep(reward, Q, t)

                    if done:
                        break

                stats.update_for_episode(info)

            stats.update_for_epoch()

            # Evaluate the agent (run the learned policy for a number of episodes)
            eval_stats = self.evaluate(episodes_to_evaluate)

            stats.print_header()
            stats.print_progress()
            stats.print()
            if (episodes_to_evaluate > 0):
                eval_stats.print()

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
        return self.env.action_space.sample()

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
        pass

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
        if n_episodes == 0:
            return

        stats = util.Stats(self.name, self.env_name, n_episodes, n_epochs=1, dt=0.02, logger=None, name = "Evaluation")
        stats.init_for_epoch()
        for episode in range(n_episodes):
            stats.init_for_episode()
            state = self.env.reset()
            episode_reward = 0

            for t in range(self.episode_horizon):
                stats.init_for_timestep()

                if (render):
                    self.env.render()

                # Select an action based on the exploration policy
                action = self.predict(state)

                # Execute the action on the environment  and observe reward and next state
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                state = next_state

                stats.update_for_timestep(reward, t)

                if done:
                    break
            stats.update_for_episode(info)
        stats.update_for_epoch()
        return stats

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
        pass

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

    def __init__(self, sess, input_dim, hidden_dims, out_dim, name = ""):
        self.sess = sess
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.name = name

        # Create online/learned network and the target network for the Actor
        self.inputs, self.out, self.net_params = self.create_architecture()

    def create_architecture(self):
        """
        Creates the architecture of the neural network. Implemented in child classes.

        Returns
        -------
        tf.Tensor
            A Tensor representing the input layer of the network.
        list of tf.Variable
            The network learnable parameters.
        tf.Tensor
            A Tensor representing the output layer of the network.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the child class.
        """
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
        return self.sess.run(self.out, feed_dict={self.inputs: inputs})

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
