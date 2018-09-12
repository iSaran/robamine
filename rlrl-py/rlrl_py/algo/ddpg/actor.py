from network import Network
import tflearn
import tensorflow as tf

class Actor(Network):
    '''Actor: Input the state, output the action
    '''
    def __init__(self, sess, state_dim, action_dim, n_units, final_layer_init, tau):
        Network.__init__(self, sess, state_dim, action_dim, n_units, final_layer_init, tau)

        # Create online/learned network and the target network for the Actor
        self.inputs, self.out = self.create_architecture()
        self.net_params = tf.trainable_variables()
        self.target_inputs, self.target_out = self.create_architecture()
        self.target_net_params = tf.trainable_variables()[len(self.net_params):]

        # Operation for updating target network with learned network weights
        self.update_target_net_params = \
            [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                                  tf.multiply(self.target_net_params[i], 1. - self.tau))
                for i in range(len(self.target_net_params))]

        # Create a placeholder for the gradient, which will be provided by the critic
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])

        self.unnormalized_actor_gradients = tf.gradients(self.out, self.net_params, -self.action_gradient)
        #self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))


    def create_architecture(self):
        # Create the input layer
        inputs = tflearn.input_data(shape=[None, self.state_dim])
        net = inputs

        # Create the hidden layers
        for n in self.n_units:
            net = tflearn.fully_connected(net, n, activation='relu')
            net = tflearn.layers.normalization.batch_normalization(net)

        # The final weights and biases are initialized from a uniform
        # distributionto ensure the initial outputs for the policy estimates is
        # near zero.
        w_init = tflearn.initializations.uniform(minval=self.final_layer_init[0], maxval=self.final_layer_init[1])

        # Final layer of the actor is a tah layer to bound the actions
        out = tflearn.fully_connected(net, self.action_dim, activation='tanh', weights_init=w_init )

        return inputs, out

    def train(self):
        pass

    def predict(self):
        pass

    def predict_target(self):
        pass
