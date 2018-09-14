from rlrl_py.algo.core import Network
import tflearn
import tensorflow as tf

class Actor(Network):
    '''Actor: Input the state, output the action
    '''
    def __init__(self, sess, input_dim, hidden_dims, out_dim, final_layer_init, batch_size, learning_rate):
        self.final_layer_init = final_layer_init
        Network.__init__(self, sess, input_dim, hidden_dims, out_dim, "Actor")

        # Create a placeholder for the gradient, which will be provided by the critic
        self.action_gradient = tf.placeholder(tf.float32, [None, self.out_dim])
        self.unnormalized_gradients = tf.gradients(self.out, self.net_params, -self.action_gradient)
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
        self.sess.run(self.optimize, feed_dict={self.inputs: inputs, self.action_gradient: a_gradient})

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
