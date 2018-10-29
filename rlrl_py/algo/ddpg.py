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

    from rlrl_py.algo.ddpg import DDPG
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
import tflearn
import tensorflow as tf

from rlrl_py.algo.core import Network, Agent
from rlrl_py.algo.util import OrnsteinUhlenbeckActionNoise

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
        self.seed(seed)

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

    def seed(self, seed):
        random.seed(seed)

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
    def __init__(self, sess, input_dim, hidden_dims, out_dim, final_layer_init, batch_size, learning_rate):
        self.final_layer_init = final_layer_init
        Network.__init__(self, sess, input_dim, hidden_dims, out_dim, "Actor")

        # Here we store the grad of Q w.r.t the actions. This gradient is
        # provided by the Critic and is used to implement the policy gradient
        # below.
        self.grad_q_wrt_a = tf.placeholder(tf.float32, [None, self.out_dim])

        # Calculate the gradient of the policy's actions w.r.t. the policy's
        # parameters and multiply it by the gradient of Q w.r.t the actions
        self.unnormalized_gradients = tf.gradients(self.out, self.net_params, -self.grad_q_wrt_a)
        self.gradients = list(map(lambda x: tf.div(x, batch_size), self.unnormalized_gradients))

        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.gradients, self.net_params))

    def create_architecture(self):
        """
        Creates the architecture of the Actor as described in Section 7 Experimental Details of :cite:`lillicrap15`.

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

        # Create the input layer
        inputs = tflearn.input_data(shape=[None, self.input_dim])
        net = inputs

        # Create the hidden layers
        for dim in self.hidden_dims:
            net = tflearn.fully_connected(net, dim, name = self.name + 'FullyConnected')
            net = tflearn.layers.normalization.batch_normalization(net, name = self.name + 'BatchNormalization')
            net = tflearn.activations.relu(net)

        # Create the output layer
        # The final weights and biases are initialized from a uniform
        # distributionto ensure the initial outputs for the policy estimates is
        # near zero, with a tah layer. The tanh layer is used for bound the
        # actions. TODO(isaran): Not sure if it is needed as long as you have
        # outputs in [-1, 1]
        w_init = tflearn.initializations.uniform(minval=self.final_layer_init[0], maxval=self.final_layer_init[1])
        out = tflearn.fully_connected(net, self.out_dim, activation='tanh', weights_init=w_init, name = self.name + 'FullyConnected')

        net_params = tf.trainable_variables()[existing_num_trainable_params:]
        return inputs, out, net_params

    def learn(self, inputs, a_gradient):
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
        self.sess.run(self.optimize, feed_dict={self.inputs: inputs, self.grad_q_wrt_a: a_gradient})

class TargetActor(Actor):
    """
    The target network of the actor.

    Parameters
    ----------
    actor : :class:`.Actor`
        The Actor
    tau : float
        A number less than 1, used to update the target's parameters
    """
    def __init__(self, actor, tau):
        self.final_layer_init = actor.final_layer_init
        Network.__init__(self, actor.sess, actor.input_dim, actor.hidden_dims, actor.out_dim, "TargetActor")

        # Operation for updating target network with learned network weights.
        self.actor_net_params = actor.net_params
        self.update_net_params = \
            [self.net_params[i].assign( \
                tf.multiply(self.actor_net_params[i], tau) + tf.multiply(self.net_params[i], 1. - tau))
                for i in range(len(self.net_params))]

        # Define an operation to set my params the same as the actor's for initialization
        self.equal_params = [self.net_params[i].assign(self.actor_net_params[i]) for i in range(len(self.net_params))]

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
    def __init__(self, sess, input_dim, hidden_dims, final_layer_init=(-0.003, 0.003), learning_rate=0.001):
        self.final_layer_init = final_layer_init
        Network.__init__(self, sess, input_dim, hidden_dims, 1, "Critic")
        assert len(self.input_dim) == 2, len(self.hidden_dims) == 2

        self.q_value = tf.placeholder(tf.float32, [None, 1])
        self.loss = tflearn.mean_square(self.q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.grad_q_wrt_actions = tf.gradients(self.out, self.inputs[1])

    def create_architecture(self):
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

        state_dim, action_dim = self.input_dim
        hidden_dim_1, hidden_dim_2 = self.hidden_dims

        # Create the input layers for state and actions
        state_inputs = tflearn.input_data(shape=[None, state_dim], name = self.name + 'InputData')
        action_inputs = tflearn.input_data(shape=[None, action_dim], name = self.name + 'InputData')

        # First hidden layer with only the states
        net = tflearn.fully_connected(state_inputs, hidden_dim_1, name = self.name + 'FullyConnected')

        net = tflearn.layers.normalization.batch_normalization(net, name = self.name + 'BatchNormalization')
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, hidden_dim_2, name = self.name + 'FullyConnected')
        t2 = tflearn.fully_connected(action_inputs, hidden_dim_2, name = self.name + 'FullyConnected')

        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action_inputs, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=self.final_layer_init[0], maxval=self.final_layer_init[1])
        out = tflearn.fully_connected(net, 1, weights_init=w_init, name = self.name + 'FullyConnected')

        net_params = tf.trainable_variables()[existing_num_trainable_params:]
        return (state_inputs, action_inputs), out, net_params

    def learn(self, inputs, q_value):
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
        return self.sess.run(self.optimize, feed_dict={self.inputs: inputs, self.q_value: q_value})

    def get_grad_q_wrt_actions(self, inputs):
        """
        Returns the gradient of the predicted Q value (output of Critic) with
        respect the actions (the second input of the Critic). To be used for
        policy improvement in Actor.

        Parameters
        ----------
        inputs : tf.Tensor
            A tuple of two minibatches of states and actions.
        """
        return self.sess.run(self.grad_q_wrt_actions, feed_dict = {self.inputs: inputs})

class TargetCritic(Critic):
    """
    The target network of the critic.

    Parameters
    ----------
    actor : :class:`.critic`
        The Critic
    tau : float
        A number less than 1, used to update the target's parameters
    """
    def __init__(self, critic, tau):
        self.final_layer_init = critic.final_layer_init
        Network.__init__(self, critic.sess, critic.input_dim, critic.hidden_dims, critic.out_dim, "TargetCritic")

        # Operation for updating target network with learned network weights.
        self.critic_net_params = critic.net_params
        self.update_net_params = \
            [self.net_params[i].assign( \
                tf.multiply(self.critic_net_params[i], tau) + tf.multiply(self.net_params[i], 1. - tau))
                for i in range(len(self.net_params))]

        # Define an operation to set my params the same as the critic's for initialization
        self.equal_params = [self.net_params[i].assign(self.critic_net_params[i]) for i in range(len(self.net_params))]


    def update_params(self):
        """
        Updates the parameters of the target network based on the current
        parameters of the critic network.
        """
        self.sess.run(self.update_net_params)

    def equalize_params(self):
        """
        Set the parameters of the target network equal to the parameters of the
        critic. Used in the initialization of the algorithm.
        """
        self.sess.run(self.equal_params)

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
    def __init__(self, sess, env, random_seed=999, log_dir='/tmp',
            replay_buffer_size=1e6, actor_hidden_units=(400, 300), final_layer_init=(-3e-3, 3e-3),
            batch_size=64, actor_learning_rate=1e-4, tau=1e-3, critic_hidden_units=(400, 300),
            critic_learning_rate=1e-3, gamma=0.999, exploration_noise_sigma=0.1):
        self.sess = sess
        self.gamma = gamma

        # Initialize the Actor network and its target net
        state_dim = int(self.env.observation_space.shape[0])
        self.action_dim = int(self.env.action_space.shape[0])
        self.actor = Actor(self.sess, state_dim, actor_hidden_units, self.action_dim, final_layer_init, batch_size, actor_learning_rate)
        self.target_actor = TargetActor(self.actor, tau)

        # Initialize the Critic network and its target net
        self.critic = Critic(self.sess, (state_dim, self.action_dim), critic_hidden_units, final_layer_init, critic_learning_rate)
        self.target_critic = TargetCritic(self.critic, tau)

        # Initialize target networks with weights equal to the learned networks
        self.sess.run(tf.global_variables_initializer())
        self.target_actor.equalize_params()
        self.target_critic.equalize_params()

        # Initialize replay buffer
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size, random_seed)

        self.exploration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim), sigma = exploration_noise_sigma)

        super(DDPG, self).__init__(sess, env, random_seed, log_dir, "DDPG")

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
        obs = state.reshape(1, state.shape[0])
        return self.actor.predict(obs).squeeze() + self.exploration_noise()

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
        return np.reshape(self.actor.predict(obs), (self.action_dim,))

    def learn(self, state, action, reward, next_state, done):
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


        self.logger.console.debug("Storing transition into replay buffer")
        self.replay_buffer.store(state, action, reward, next_state, done)
        self.logger.console.debug('Size of replay buffer: ' + str(self.replay_buffer.size()))

        # If we have not enough samples just keep storing transitions to the
        # buffer and thus exit.
        if self.replay_buffer.size() < self.batch_size:
            return

        self.logger.console.debug("Sampling a minibatch from the replay buffer")
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.replay_buffer.sample_batch(self.batch_size)

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

        self.logger.console.debug("Evaluating policy")
        mu_target_next = self.target_actor.predict(next_state_batch)
        Q_target_next = self.target_critic.predict((next_state_batch, mu_target_next))

        y_i = []
        for k in range(self.batch_size):
            if terminal_batch[k]:
                y_i.append(reward_batch[k])
            else:
                y_i.append(reward_batch[k] + self.gamma * Q_target_next[k])

        y_i = np.reshape(y_i, (64, 1))

        self.critic.learn((state_batch, action_batch), y_i)

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

        self.logger.console.debug("Improving policy")
        mu = self.actor.predict(state_batch)
        grads = self.critic.get_grad_q_wrt_actions((state_batch, mu))
        self.actor.learn(state_batch, grads[0])

    def q_value(self, state, action):
        return self.critic.predict((np.reshape(state, (1, state.shape[0])), np.reshape(action, (1, action.shape[0])))).squeeze()

    def seed(self, seed):
        super(DDPG, self).seed(seed)
        self.exploration_noise.seed(seed)
        self.replay_buffer.seed(seed)
