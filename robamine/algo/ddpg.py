"""
Deep Deterministic Policy Gradient
==================================

This module contains the implementation of the Deep Deterministic Policy
Gradient algorithm (DDPG) :cite:`lillicrap15`. For the details of the algorithm
please read the paper.

The basic class is :class:`.DDPG`. All the other classes in this module
implement various components of DDPG and are used by this main class. The other
components are the Actor, the Critic (and their target networks) and the
replay buffer.

Example
-------

This is a minimal example that trains DDPG for the environment ``MyRobot``, for 1000 episodes:

.. code-block:: python

    from robamine.algo.ddpg import DDPG
    import tensorflow as tf
    with tf.Session as session:
        agent = DDPG(session, 'MyRobot').train(1000)

Or you can use parameters different from the defaults:

.. code-block:: python

    with tf.Session as session:
        agent = DDPG(session, 'MyRobot', actor_learning_rate=0.0002, critic_learning_rate=0.001).train(1000)
"""

from collections import deque
import random
import numpy as np
import tensorflow as tf
import gym
import pickle
import os

from robamine.algo.core import Network, NetworkParams, Agent, AgentParams
from robamine.algo.util import OrnsteinUhlenbeckActionNoise
import math

import logging

logger = logging.getLogger('robamine.algo.ddpg')

class ReplayBuffer:
    """
    Implementation of the replay experience buffer. Creates a buffer which uses
    the deque data structure. Here you can store experience transitions (i.e.: state,
    action, next state, reward) and sample mini-batches for training.

    You can  retrieve a transition like this:

    Example of use:

    .. code-block:: python

        replay_buffer = ReplayBuffer(10)
        replay_buffer.store()
        replay_buffer.store([0, 2, 1], [1, 2], -12.9, [2, 2, 1], 0)
        # ... more storing
        transition = replay_buffer(2)


    Parameters
    ----------
    buffer_size : int
        The buffer size
    seed : int, optional
        A seed for initializing the random batch sampling and have reproducable
        results.
    """
    def __init__(self, buffer_size, seed=999):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    def __call__(self, index):
        """
        Returns a transition from the buffer.

        Parameters
        ----------
        index : int
            The index number of the desired transition

        Returns
        -------
        tuple
            The transition

        """
        return self.buffer[index]

    def store(self, state, action, reward, next_state, terminal):
        """
        Stores a new transition on the buffer.

        Parameters
        ----------
        state : np.ndarray
            The state of the transition
        action : np.ndarray
            The action of the transition
        reward : np.float32
            The reward of the transition
        next_state : np.ndarray
            The next state of the transition
        terminal : np.float32
            1 if this state is terminal. 0 otherwise.
        """
        transition = (state, action, reward, next_state, terminal)
        if self.count < self.buffer_size:
            self.count += 1
        else:
            self.buffer.popleft()
        self.buffer.append(transition)

    def sample_batch(self, given_batch_size):
        """
        Samples a minibatch from the buffer.

        Parameters
        ----------
        given_batch_size : int
            The size of the mini-batch.

        Returns
        -------
        numpy.array
            The state batch
        numpy.array
            The action batch
        numpy.array
            The reward batch
        numpy.array
            The next state batch
        numpy.array
            The terminal batch
        """
        batch = []

        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch = random.sample(self.buffer, batch_size)

        state_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        reward_batch = np.array([_[2] for _ in batch])
        next_state_batch = np.array([_[3] for _ in batch])
        terminal_batch = np.array([_[4] for _ in batch])

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def clear(self):
        """
        Clears the buffer my removing all elements.
        """
        self.buffer.clear()
        self.count = 0

    def size(self):
        """
        Returns the current size of the buffer.

        Returns
        -------
        int
            The number of existing transitions.
        """
        return self.count

    def seed(self, random_seed):
        random.seed(random_seed)

class ActorParams(NetworkParams):
    def __init__(self,
                 hidden_units = (400, 300),
                 name = "Actor",
                 learning_rate = 1e-4,
                 gate_gradients = False,
                 final_layer_init=(-3e-3, 3e-3),
                 batch_size = 64):
        super().__init__(hidden_units=hidden_units,
                         name=name)
        self.learning_rate = learning_rate
        self.gate_gradients = gate_gradients
        self.final_layer_init = final_layer_init
        self.batch_size = batch_size

class Actor(Network):
    """
    Implements the Actor.

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
        The dimensions of the state space.
    hidden_dims : tuple
        The dimensions of each hidden layer. The size of tuple defines the
        number of the hidden layers.
    out_dim : int
        The dimensions of the action space.
    final_layer_init : tuple of 2 int
        The range to randomly initialize the parameters of the final layer in
        order to have action predictions close to zero in the beginning.
    batch_size : int
        The size of the mini-batch that is being used (in order to normalize
        the gradients)
    learning_rate : float
        The learning rate for the optimizer
    """
    def __init__(self, sess, params):
        super().__init__(sess, params)
        self.state_input = None

    @classmethod
    def create(cls, sess, params):
        self = cls(sess, params)

        state_input_dim = params.input_dim
        with tf.variable_scope(self.params.name):
            self.state_input = tf.placeholder(tf.float32, [None, state_input_dim], name='state_input')
            with tf.variable_scope('network'):
                self.out, self.trainable_params = self.architecture(self.state_input, self.params.hidden_units, self.params.output_dim, self.params.final_layer_init)

            # Here we store the grad of Q w.r.t the actions. This gradient is
            # provided by the Critic and is used to implement the policy gradient
            # below.
            self.grad_q_wrt_a = tf.placeholder(tf.float32, [None, self.params.output_dim], name='gradQ_a')

            # Calculate the gradient of the policy's actions w.r.t. the policy's
            # parameters and multiply it by the gradient of Q w.r.t the actions
            unnormalized_gradients = tf.gradients(self.out, self.trainable_params, -self.grad_q_wrt_a, gate_gradients=self.params.gate_gradients, name='sum_gradQa_grad_mutheta')
            gradients = list(map(lambda x: tf.div(x, self.params.batch_size, name='div_by_N'), unnormalized_gradients))
            self.optimizer = tf.train.AdamOptimizer(self.params.learning_rate, name='optimizer').apply_gradients(zip(gradients, self.trainable_params))

        return self

    @staticmethod
    def architecture(state_input, hidden_dims, out_dim, final_layer_init):
        # Check the number of trainable params before you create the network
        existing_num_trainable_params = len(tf.trainable_variables())

        # Input layer
        net = state_input
        state_input_dim = state_input.get_shape().as_list()[1]
        fan_in = state_input_dim  # TODO: this seems wrong, use every layers input, but is not critical
        for dim in hidden_dims:
            weight_initializer = tf.initializers.random_uniform(minval= - 1 / math.sqrt(fan_in), maxval = 1 / math.sqrt(fan_in))
            net = tf.layers.dense(inputs=net, units=dim, kernel_initializer=weight_initializer, bias_initializer=weight_initializer)
            net = tf.layers.batch_normalization(inputs=net)
            net = tf.nn.relu(net)

        # Final layer
        weight_initializer = tf.initializers.random_uniform(minval=final_layer_init[0], maxval=final_layer_init[1])
        net = tf.layers.dense(inputs=net, units=out_dim, kernel_initializer= weight_initializer, bias_initializer= weight_initializer)

        out = tf.nn.tanh(net)
        net_params = tf.trainable_variables()[existing_num_trainable_params:]
        return out, net_params

    def learn(self, state, a_gradient):
        """
        Trains the neural network, using the gradient of the Q value with
        respect to actions, which is provided by Critic. This gradient is
        multiplied with the gradient of the actions with respect to the network
        parameters and the final result is applied as the gradient to optimize
        the Actor with Adam optimization. Thus, it optimizes the Actor's
        network by implementing the policy gradient (Eq. (6) of
        :cite:`lillicrap15`). These are implemented as Tensorflow operations.

        Parameters
        ----------
        inputs : tf.Tensor
            A minibatch of states
        a_gradient : tf.Tensor
            The gradient of the Q value with respect to the actions. This is provided by Critic.
        """
        if self.optimizer is None:
            logger.error('Actor network is not created or initialized.')

        self.sess.run(self.optimizer, feed_dict={self.state_input: state, self.grad_q_wrt_a: a_gradient})

    def predict(self, state):
        return self.sess.run(self.out, feed_dict={self.state_input: state})

    def to_dict(self):
        data = super(DDPG, self).to_dict()
        data['final_layer_init'] = self.final_layer_init
        data['batch_size'] = self.batch_size
        data['learning_rate'] = self.learning_rate
        return data

    @classmethod
    def from_dict(cls, data):
        net = cls(data['input_dim'], data['hidden_dims'], data['out_dim'], data['final_layer_init'], data['batch_size'], data['learning_rate'])
        net.inputs, net.out, net.net_params = data['inputs'], data['out'], data['net_params']

        # Calculate the gradient of the policy's actions w.r.t. the policy's
        # parameters and multiply it by the gradient of Q w.r.t the actions
        net.unnormalized_gradients = tf.gradients(net.out, net.net_params, -net.grad_q_wrt_a)
        net.gradients = list(map(lambda x: tf.div(x, net.batch_size), net.unnormalized_gradients))
        net.optimize = tf.train.AdamOptimizer(net.learning_rate).apply_gradients(zip(net.gradients, net.net_params))
        return net

class Target(Network):
    """
    The target network of the actor.

    Parameters
    ----------
    actor : :class:`.Actor`
        The Actor
    tau : float
        A number less than 1, used to update the target's parameters
    """
    def __init__(self, base, tau, name=None):
        assert isinstance(base, Actor) or isinstance(base, Critic), 'The base network should either an Actor or a Critic'

        n = base.params.name + '_target'
        if name is not None:
            n = n + '_' + name

        params = NetworkParams(input_dim=base.params.input_dim, hidden_units=base.params.hidden_units, output_dim=base.params.output_dim, name=n)
        super().__init__(base.sess, params)

        self.base_is_actor = False
        if isinstance(base, Actor):
            self.base_is_actor = True

        # Operation for updating target network with learned network weights.
        self.trainable_params = self.trainable_params

        self.state_input = None
        self.action_input = None

        self.update_net_params = None
        self.equal_params = None

    @classmethod
    def create(cls, base, tau):
        self = cls(base, tau)

        with tf.variable_scope(self.params.name):
            if isinstance(base, Actor):
                self.state_input = tf.placeholder(tf.float32, [None, base.params.input_dim], name='state_input')
            elif isinstance(base, Critic):
                self.state_input = tf.placeholder(tf.float32, [None, base.params.input_dim[0]], name='state_input')
                self.action_input = tf.placeholder(tf.float32, [None, base.params.input_dim[1]], name='action_input')

            with tf.variable_scope('network'):

                if isinstance(base, Actor):
                    self.out, self.trainable_params = Actor.architecture(self.state_input, base.params.hidden_units, base.params.output_dim, base.params.final_layer_init)
                elif isinstance(base, Critic):
                    self.out, self.trainable_params = Critic.architecture(self.state_input, self.action_input, base.params.hidden_units, base.params.final_layer_init)

            with tf.variable_scope('update_params_with_base'):
                self.update_net_params = \
                    [self.trainable_params[i].assign( \
                        tf.multiply(base.trainable_params[i], tau) + tf.multiply(self.trainable_params[i], 1. - tau))
                        for i in range(len(self.trainable_params))]

            # Define an operation to set my params the same as the actor's for initialization
            with tf.variable_scope('set_params_equal_to_base'):
                self.equal_params = [self.trainable_params[i].assign(base.trainable_params[i]) for i in range(len(base.trainable_params))]

        return self

    def update_params(self):
        """
        Updates the parameters of the target network based on the current
        parameters of the actor network.
        """
        self.sess.run(self.update_net_params)

    def equalize_params(self):
        """
        Set the parameters of the target network equal to the parameters of the
        actor. Used in the initialization of the algorithm.
        """
        self.sess.run(self.equal_params)

    def predict(self, state, action = None):
        if self.base_is_actor:
            return self.sess.run(self.out, feed_dict={self.state_input: state})
        else:
            return self.sess.run(self.out, feed_dict={self.state_input: state, self.action_input: action})

class CriticParams(NetworkParams):
    def __init__(self,
                 hidden_units = (400, 300),
                 name = "Critic",
                 learning_rate = 1e-4,
                 final_layer_init=(-3e-3, 3e-3)):

        super().__init__(hidden_units=hidden_units,
                         name=name)
        self.learning_rate = learning_rate
        self.final_layer_init = final_layer_init
        self.output_dim = 1

class Critic(Network):
    """
    Implements the Critic.

    Attributes
    ----------
    inputs : tf.Tensor
        A Tensor representing the input layer of the network.
    out : tf.Tensor
        A Tensor representing the output layer of the network.
    net_params : list of tf.Variable
        The network learnable parameters.
    input_dim : tuple of 2 ints
        A tuple with the dimensions of the states and the actions
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
    input_dim : tuple of two ints
        A tuple with the dimensions of the states and the actions
    hidden_dims : tuple
        The dimensions of each hidden layer. The size of tuple defines the
        number of the hidden layers.
    out_dim : int
        The dimensions of the action space.
    final_layer_init : tuple of 2 int
        The range to randomly initialize the parameters of the final layer in
        order to have action predictions close to zero in the beginning.
    learning_rate : float
        The learning rate for the optimizer
    """
    def __init__(self, sess, params=CriticParams()):
        super().__init__(sess, params)
        self.state_dim = self.params.input_dim[0]
        self.action_dim = self.params.input_dim[1]
        self.state_input = None
        self.action_input = None

    @classmethod
    def create(cls, sess, params):
        self = cls(sess, params)

        with tf.variable_scope(self.params.name):
            self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], name='state_input')
            self.action_input = tf.placeholder(tf.float32, [None, self.action_dim], name='action_input')
            with tf.variable_scope('network'):
                self.out, self.trainable_params = self.architecture(self.state_input, self.action_input, self.params.hidden_units, self.params.final_layer_init)

            self.q_value = tf.placeholder(tf.float32, [None, 1], name='Q_value')
            self.loss = tf.losses.mean_squared_error(self.q_value, self.out)
            self.optimizer = tf.train.AdamOptimizer(self.params.learning_rate, name='optimize_mse').minimize(self.loss)

            # # Get the gradient of the net w.r.t. the action.
            # # For each action in the minibatch (i.e., for each x in xs),
            # # this will sum up the gradients of each critic output in the minibatch
            # # w.r.t. that action. Each output is independent of all
            # # actions except for one.
            self.grad_q_wrt_actions = tf.gradients(self.out, self.action_input, name='gradQ_a')

        return self

    @staticmethod
    def architecture(state, action, hidden_units, final_layer_init):
        """
        Creates the architecture of the Critic as described in Section 7 Experimental Details of :cite:`lillicrap15`.

        Returns
        -------
        tf.Tensor
            A Tensor representing the input layer of the network.
        list of tf.Variable
            The network learnable parameters.
        tf.Tensor
            A Tensor representing the output layer of the network.
        """
        # Check the number of trainable params before you create the network
        existing_num_trainable_params = len(tf.trainable_variables())

        state_dim, action_dim = state.get_shape().as_list()[1], action.get_shape().as_list()[1]
        hidden_units_1, hidden_units_2 = hidden_units

        fan_in = fan_in = state_dim  # TODO: this seems wrong, use every layers input, but is not critical
        weight_initializer = tf.initializers.random_uniform(minval= - 1 / math.sqrt(fan_in), maxval = 1 / math.sqrt(fan_in))

        # First hidden layer with only the states
        net = tf.layers.dense(inputs=state, units=hidden_units_1, kernel_initializer=weight_initializer, bias_initializer=weight_initializer)
        net = tf.layers.batch_normalization(inputs=net)
        net = tf.nn.relu(net)

        # Second layer with actions
        action_and_first_layer = tf.concat([net, action], axis=1)
        net = tf.layers.dense(inputs=action_and_first_layer, units=hidden_units_2, kernel_initializer=weight_initializer, bias_initializer=weight_initializer)
        net = tf.layers.batch_normalization(inputs=net)
        net = tf.nn.relu(net)

        # Output layer
        weight_initializer = tf.initializers.random_uniform(minval=final_layer_init[0], maxval=final_layer_init[1])
        net = tf.layers.dense(inputs=net, units=1, kernel_initializer=weight_initializer, bias_initializer=weight_initializer)

        out = net
        net_params = tf.trainable_variables()[existing_num_trainable_params:]
        return out, net_params

    def learn(self, state, action, q_value):
        """
        Trains the neural network, using the target Q value provided by the
        target network.

        Parameters
        ----------
        inputs : tf.Tensor
            A tuple of two minibatches of states and actions.
        q_value : tf.Tensor
            The target Q value.
        """
        return self.sess.run(self.optimizer, feed_dict={self.state_input: state, self.action_input: action, self.q_value: q_value})

    def predict(self, state, action):
        return self.sess.run(self.out, feed_dict={self.state_input: state, self.action_input: action})

    def get_grad_q_wrt_actions(self, state, action):
        """
        Returns the gradient of the predicted Q value (output of Critic) with
        respect the actions (the second input of the Critic). To be used for
        policy improvement in Actor.

        Parameters
        ----------
        inputs : tf.Tensor
            A tuple of two minibatches of states and actions.
        """
        return self.sess.run(self.grad_q_wrt_actions, feed_dict={self.state_input: state, self.action_input: action})

class DDPGParams(AgentParams):
    def __init__(self,
                 random_seed=999,
                 suffix="",
                 replay_buffer_size = 1e6,
                 batch_size = 64,
                 gamma = 0.999,
                 exploration_noise_sigma = 0.2,
                 tau = 1e-3,
                 actor = ActorParams(),
                 critic = CriticParams()):
        super().__init__(random_seed, "DDPG", suffix)

        # DDPG params
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_noise_sigma = exploration_noise_sigma
        self.tau = tau
        self.actor = actor
        self.critic = critic

class DDPG(Agent):
    """
    Implements the DDPG algorithm.

    Parameters
    ----------
    sess : :class:`.tf.Session`
    env : str
        A string with the name of a registered Gym Environment
    random_seed : int, optional
        A random seed for reproducable results.
    log_dir : str
        A directory for storing the trained model and logged data.
    replay_buffer_size : int
        The size of the replay buffer.
    actor_hidden_units : tuple
        The number of the units for the hidden layers for the actor.
    final_layer_init : tuple of 2 int
        The range to randomly initialize the parameters of the final layers of actor and critic in
        order to have action predictions close to zero in the beginning.
    batch_size : int
        The size of the minibatch
    actor_learning_rate : float
        The learning rate for the actor.
    tau : float
        A number less than 1, used to update the target's parameters for both actor and critic
    critic_hidden_units : tuple
        The number of the units for the hidden layers for the critic.
    critic_learning_rate : float
        The learning rate of the critic.
    gamma : float
        The discounting factor
    exploration_noise_sigma : float
        The sigma for the OrnsteinUhlenbeck Noise for exploration.
    """
    def __init__(self, sess, params):

        super().__init__(sess, params)

        self.actor = None
        self.target_actor = None
        self.critic = None
        self.target_critic = None
        self.replay_buffer = None
        self.exploration_noise = None

    @classmethod
    def create(cls, sess, params):
        self = cls(sess, params)
        # Initialize the Actor network and its target net
        actor_params = params.actor
        actor_params.input_dim = self.params.state_dim
        actor_params.output_dim = self.params.action_dim
        self.actor = Actor.create(self.sess, actor_params)
        self.target_actor = Target.create(self.actor, self.params.tau)

        # Initialize the Critic network and its target net
        critic_params = params.critic
        critic_params.input_dim = (self.params.state_dim, self.params.action_dim)
        self.critic = Critic.create(self.sess, critic_params)
        self.target_critic = Target.create(self.critic, self.params.tau)

        # Initialize target networks with weights equal to the learned networks
        self.sess.run(tf.global_variables_initializer())
        self.target_actor.equalize_params()
        self.target_critic.equalize_params()

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.params.replay_buffer_size, self.params.random_seed)

        self.exploration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.params.action_dim), sigma = self.params.exploration_noise_sigma)

        return self

    def explore(self, state):
        """
        Represents the exploration policy which is the predictions of the Actor
        (the learned policy) plus OrnsteinUhlenbeck noise.

        Parameters
        ----------

        state : numpy array
            The current state of the environment.

        Returns
        -------
        numpy array
            An action to be performed for exploration.
        """
        obs = np.array(state.reshape(1, state.shape[0]))
        prediction = self.actor.predict(obs).squeeze()
        noise = self.exploration_noise()
        logger.debug('Explore: prediction:' + prediction.__str__())
        logger.debug('DDPG explore: noise:' + noise.__str__())
        return prediction + noise

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
        obs = state.reshape(1, state.shape[0])
        return np.reshape(self.actor.predict(obs), (self.params.action_dim,))

    def learn(self, transition):
        """
        Implements the DDPG for each timestep of the episode:

            * Stores the transition in the replay buffer and samples a minibatch from it.
            * Policy evaluation (see :meth:`.evaluate_policy`)
            * Policy improvement (see :meth:`.improve_policy`)
            * Update target networks.

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


        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state
        done = transition.terminal

        logger.debug("Storing transition into replay buffer")
        self.replay_buffer.store(state, action, reward, next_state, done)
        logger.debug('Size of replay buffer: %d', self.replay_buffer.size())

        # If we have not enough samples just keep storing transitions to the
        # buffer and thus exit.
        if self.replay_buffer.size() < self.params.batch_size:
            return

        logger.debug("Sampling a minibatch from the replay buffer")
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.replay_buffer.sample_batch(self.params.batch_size)
        logger.debug("Sampled batches by replay buffer:")
        logger.debug('State batch: %s', state_batch.__str__())
        logger.debug('Action batch: %s', action_batch.__str__())
        logger.debug('Reward batch: %s', reward_batch.__str__())
        logger.debug('next state batch: %s', next_state_batch.__str__())
        logger.debug('terminal batch: %s', reward_batch.__str__())

        self.evaluate_policy(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
        self.improve_policy(state_batch)

        self.target_actor.update_params()
        self.target_critic.update_params()

    def evaluate_policy(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        """
        Performs policy evaluation, i.e. trains the Critic. This means that it
        calculates the target Q values :math:`y_i`, using the target Actor and
        critic and use them in loss to train the Critic. :

            * Calculates the target Q value batch :math:`y_i`:

                * Feeds the next state batch to the target actor and calculates a next action batch :math:`a = \mu^\prime(s_{i+1} | \\theta^{\mu^{\prime}})`
                * Feeds the next action batch and the next state batch to the target Critic for calculating :math:`Q^\prime(s_{i+1}, a)`.
                * Uses the target Q and the reward for calculating :math:`y_i`

            * Train critic by  minimizing the loss (mean square between the critic predictions and the targets :math:`y_i`)

        Parameters
        ----------
        state_batch : np.ndarray
            The state batch
        action_batch : np.ndarray
            The action batch
        reward_batch : float
            The reward batch
        next_state_batch : np.ndarray
            The batch of the next state
        terminal_batch : float
            The batch of the terminal states
        """

        logger.debug("Evaluating policy")
        mu_target_next = self.target_actor.predict(next_state_batch)
        Q_target_next = self.target_critic.predict(next_state_batch, mu_target_next)

        logger.debug('mu_target_next: %s', mu_target_next.__str__())
        logger.debug('Q_target_next: %s', Q_target_next.__str__())

        y_i = []
        for k in range(self.params.batch_size):
            if terminal_batch[k]:
                y_i.append(reward_batch[k])
            else:
                y_i.append(reward_batch[k] + self.params.gamma * Q_target_next[k])

        y_i = np.reshape(y_i, (64, 1))

        logger.debug('y_i: %s', y_i.__str__())

        self.critic.learn(state_batch, action_batch, y_i)

    def improve_policy(self, state_batch):
        """
        Performs policy improvement, i.e. trains the Actor using the gradient of the Q value from the Critic

            * Calculate :math:`\\nabla _a Q` from the output of the critic.
            * Apply the policy gradient to optimize actor

        Parameters
        ----------
        state_batch : np.ndarray
            The state batch
        """

        logger.debug("Improving policy")
        mu = self.actor.predict(state_batch)
        logger.debug('mu: %s', mu.__str__())
        grads = self.critic.get_grad_q_wrt_actions(state_batch, mu)
        logger.debug('grads: %s', grads.__str__())
        self.actor.learn(state_batch, grads[0])
        logger.debug('Actor params after learning: %s', self.actor.get_params().__str__())

    def q_value(self, state, action):
        return self.critic.predict(np.reshape(state, (1, state.shape[0])), np.reshape(action, (1, action.shape[0]))).squeeze()

    def seed(self, seed):
        super().seed(seed)
        self.exploration_noise.seed(seed)

    def save(self):
        store_file = os.path.join(self.logger.get_dir(), 'agent.pkl')
        self.logger.console.info('Saving agent to ' + store_file)
        data = {}
        data['replay_buffer'] = self.replay_buffer
        data['noise'] = self.exploration_noise
        data['random_seed'] = self.random_seed
        data['actor'] = self.actor
        with open(store_file, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    def load(self, path_to_file):
        with open(path_to_file, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        self.replay_buffer = data['replay_buffer']
        self.exploration_noise = data['exploration_noise']
        self.random_seed = data['random_seed']
        self.seed(self.random_seed)
