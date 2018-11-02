#!/usr/bin/env python3
from robamine.envs.sphere_reacher import SphereReacher
import numpy as np

class SphereReacherIncremental(SphereReacher):
    def __init__(self, distance_threshold = 0.005, target_range = 0.2, num_substeps = 10):
        SphereReacher.__init__(self, distance_threshold, target_range, num_substeps)

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        commanded = np.zeros(6)
        commanded[0] = self.map_to_new_range(action[0], [-1, 1], [-3, 3])
        commanded[1] = self.map_to_new_range(action[1], [-1, 1], [-3, 3])

        dt = self.model.opt.timestep * self.n_substeps
        commanded = self.last_commanded + dt * commanded

        self.send_applied_wrench(commanded, 'finger')

        self.last_commanded = commanded

    def _get_obs(self):
        finger_pos = self.sim.data.get_body_xpos('optoforce')
        addr = self.model.get_joint_qvel_addr("finger_free_joint")
        finger_vel = self.sim.data.qvel[addr[0]:addr[1]][:3]

        observation = np.concatenate((self.last_commanded, finger_pos, finger_vel, self.goal))
        obs = { 'observation': observation,
                'achieved_goal': finger_pos,
                'desired_goal': self.goal
              }
        return obs
