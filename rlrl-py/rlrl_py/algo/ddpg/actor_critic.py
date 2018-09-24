from rlrl_py.algo.core import Network
import tflearn
import tensorflow as tf

class Actor(Network):
    '''Actor: Input the state, output the action
    '''
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

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={self.inputs: inputs, self.grad_q_wrt_a: a_gradient})

    def get_action(self):
        return self.sess.run(self.target)

class TargetActor(Actor):
    '''Actor: Input the state, output the action
    '''
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
        self.sess.run(self.update_net_params)

    def equalize_params(self):
        self.sess.run(self.equal_params)

class Critic(Network):
    def __init__(self, sess, input_dim, hidden_dims, final_layer_init=(-0.003, 0.003), learning_rate=0.001):
        """
        Contructs a Critic Network.

        Parameters
        ----------
        sess : A Tensorflow session
        input_dim : tuple
            A tuple with the dimensions of the states and the actions
        hidden_dims: list of int
            A list with the number of units of the hidden layers. Each member
            of the list corresponds to a hidden layer.
        final_layer_init: list of size 2
            The range to which the parameters of the final layer will be initialized
        learning_rate: float
            The learning rate for the Adam optimizer. Defaults to 0.001.
        """
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

    def train(self, inputs, q_value):
        return self.sess.run(self.optimize, feed_dict={self.inputs: inputs, self.q_value: q_value})

    def get_grad_q_wrt_actions(self, inputs):
        return self.sess.run(self.grad_q_wrt_actions, feed_dict = {self.inputs: inputs})

class TargetCritic(Critic):
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
        self.sess.run(self.update_net_params)

    def equalize_params(self):
        self.sess.run(self.equal_params)
