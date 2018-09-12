class Network:
    def __init__(self, sess, state_dim, action_dim, n_units, final_layer_init, tau):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_units = n_units
        self.final_layer_init = final_layer_init
        self.tau = tau

    def create_architecture(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def predict_target(self):
        pass
