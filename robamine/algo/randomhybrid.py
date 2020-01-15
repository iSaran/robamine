from robamine.algo.core import RLAgent
from robamine.algo.util import NormalNoise
import numpy as np

default_params = {
    'actions' : [3, 2],
    'noise' : {
        'name' : 'Normal',
        'sigma' : 0.2
    }
}

class RandomHybrid(RLAgent):
    def __init__(self, state_dim, action_dim, params = default_params):
        super(RandomHybrid, self).__init__(state_dim, action_dim, 'RandomHybrid')
        self.params = params
        self.rng = np.random.RandomState()
        self.actions = params['actions']
        print(self.params)

        self.exploration_noise = []
        for i in range(len(self.actions)):
            self.exploration_noise.append(NormalNoise(mu=np.zeros(self.actions[i]), sigma=self.params['noise']['sigma']))

    def explore(self, state):
        output = np.zeros(max(self.actions))
        i = self.rng.randint(0, len(self.actions))
        output = np.zeros(max(self.actions) + 1)
        action = self.exploration_noise[i]()
        output[0] = i
        output[1:(self.actions[i] + 1)] = action
        return output

    def predict(self, state):
        return self.explore(state)

    def learn(self, transition):
        pass

    def q_value(self, state, action):
        return 0.0

    def seed(self, seed):
        self.rng.seed(seed)

    def save(self, path):
        pass
