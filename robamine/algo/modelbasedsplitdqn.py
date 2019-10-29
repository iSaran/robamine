"""
Deep Q-Network
==============
"""
from robamine.algo.splitdqn import SplitDQN
from robamine.algo.splitdynamicsmodelpose import SplitDynamicsModelPose
from robamine.algo.splitdynamicsmodelfeature import SplitDynamicsModelFeature
import numpy as np
import math

import logging
logger = logging.getLogger('robamine.algo.modelbasedsplitdqn')

class ModelBasedSplitDQN(SplitDQN):
    def __init__(self, state_dim, action_dim, params = {}):
        super().__init__(state_dim, action_dim, params)
        self.dynamics_model_pose = SplitDynamicsModelPose.load(params['dynamics_model_pose'])
        self.dynamics_model_feature = SplitDynamicsModelFeature.load(params['dynamics_model_feature'])

    def predict(self, state):
        predictions = []
        x = None
        u = None
        p = None

        print('ModelBasedSplitDQN: ', 'Predicting:', self.params['prediction_horizon'])

        for t in range(self.params['prediction_horizon']):

            if t == 0:
                x = state
                p = np.array([0, 0, 0])
            else:
                # Feature prediction
                x_pred = self.dynamics_model_feature.predict(x, u)
                x = x + x_pred

                # Pose prediction
                j = int(u - np.floor(u / self.nr_substates) * self.nr_substates)
                features = np.split(state, self.nr_substates)
                push_distance = features[j][-2]
                p_pred = self.dynamics_model_pose.predict(push_distance, u)
                p = p + p_pred

            u = super().predict(x)

            predictions.append({'action': u, 'pose': p})

        if len(predictions) == 1:
            return predictions[0]

        return predictions
