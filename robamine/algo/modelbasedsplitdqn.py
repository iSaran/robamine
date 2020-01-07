"""
Deep Q-Network
==============
"""
from robamine.algo.splitdqn import SplitDQN
from robamine.algo.splitdynamicsmodel import SplitDynamicsModel
import numpy as np

import logging
logger = logging.getLogger('robamine.algo.modelbasedsplitdqn')

class ModelBasedSplitDQN(SplitDQN):
    def __init__(self, state_dim, action_dim, params = {}):
        super().__init__(state_dim, action_dim, params)
        self.dynamics_model = SplitDynamicsModel.load(params['dynamics_model'])

    def predict(self, state):
        predictions = []
        x = None
        u = None
        p = None

        for t in range(self.params['prediction_horizon']):

            if t == 0:
                x = state
                p = np.array([0, 0, 0])
            else:
                x_pred, p_pred = self.dynamics_model.predict(x, u)
                x = x + x_pred
                p = p + p_pred

            u = super().predict(x)

            predictions.append({'action': u, 'pose': p})

        if len(predictions) == 1:
            return predictions[0]

        return predictions
