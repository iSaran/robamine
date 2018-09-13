class Network:
    '''Abstract class which defines an interface for e Neural Network
    '''
    def __init__(self, sess, input_dim, hidden_dims, out_dims):
        self.sess = sess
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dims

        # Create online/learned network and the target network for the Actor
        self.inputs, self.out, self.net_params = self.create_architecture()

    def create_architecture(self):
        raise NotImplementedError

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={self.inputs: inputs})
