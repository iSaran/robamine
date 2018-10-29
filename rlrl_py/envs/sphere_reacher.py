"""
Sphere Reacher
==============
"""
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer

from mujoco_py import functions
from mujoco_py.generated import const

from gym import utils, spaces
from gym.envs.robotics import robot_env
import os

import rlrl_py.utils as arl
import random

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class SphereReacher(robot_env.RobotEnv, utils.EzPickle):
    def __init__(self, distance_threshold = 0.005, target_range = 0.2, num_substeps = 10, shaped = False):

        self.distance_threshold = distance_threshold
        self.target_range = target_range
        self.shaped = shaped
        # Create MuJoCo Model
        path = os.path.join(os.path.dirname(__file__),
                            "assets/xml/robots/sphere_reacher.xml")
        self.model = load_model_from_path(path)
        self.first_time = True

        self.last_commanded = np.zeros(shape=(6,))
        self.init_opto_pos = [0, 0, 0.033]
        self.n_actions = 2
        init_qpos = {}
        self.n_substeps = num_substeps
        robot_env.RobotEnv.__init__(self, path, init_qpos, n_actions=self.n_actions, n_substeps=num_substeps)
        utils.EzPickle.__init__(self)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = goal_distance(achieved_goal, desired_goal)
        if self.shaped:
            return -distance**
        return -(distance > self.distance_threshold).astype(np.float32)

    # RobotEnv methods
    # ----------------------------

    def _is_success(self, achieved_goal, desired_goal):
        distance = goal_distance(achieved_goal, desired_goal)
        return (distance < self.distance_threshold).astype(np.float32)

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        commanded = np.zeros(6)
        commanded[0] = action[0]
        commanded[1] = action[1]

        commanded = 3 * commanded

        applied_force = commanded
        self.send_applied_wrench(applied_force, 'finger')

    def _get_obs(self):
        # Calculate the distance betwe
        finger_pos = self.sim.data.get_body_xpos('optoforce')
        addr = self.model.get_joint_qvel_addr("finger_free_joint")
        finger_vel = self.sim.data.qvel[addr[0]:addr[1]][:3]
        observation = np.concatenate((finger_pos, finger_vel, self.goal))

        obs = { 'observation': observation,
                'achieved_goal': finger_pos,
                'desired_goal': self.goal
              }
        return obs

    def _sample_goal(self):
        "Samples a goal for the target"
        random = np.zeros(shape=(3,))
        while np.linalg.norm(random) < 2 * self.distance_threshold:
            random = np.random.uniform(-self.target_range, self.target_range, 3)
            random[2] = 0.0
        return self.init_opto_pos +  random


    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('optoforce')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        # Set the camera configuration (spherical coordinates)
        self.viewer.cam.distance = 0.846
        self.viewer.cam.elevation = -21.774
        self.viewer.cam.azimuth = -148.306

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        self.disturbance = np.random.normal(2, 1, 2)
        self.disturbance[0] = random.choice([-1, 1]) * self.disturbance[0]
        self.disturbance = [0, 0]
        return True

    def _render_callback(self):
        if self.first_time:
            self.init_goal = self.goal
            self.first_time = False
        site_id = self.sim.model.site_name2id('target')
        self.sim.model.site_pos[site_id] = self.goal[0:3]
        self.sim.forward()

    def terminal_state(self, observation):
        if np.linalg.norm(observation[0:3]) > 0.5:
            return True

        return False

    def get_contact_force(self, geom1_name, geom2_name):
        result = np.zeros(shape=6)
        have_contact = False
        for i in range(0, self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if (self.model.geom_id2name(contact.geom1) == geom1_name) and (self.model.geom_id2name(contact.geom2) == geom2_name):
                functions.mj_contactForce(self.model, self.sim.data, i, result)
                frame = np.reshape(contact.frame, (-1, 3))
                result[0:3] = np.matmul(frame, result[0:3])
        return result[0:3]

    def map_to_new_range(self, value, range_old, range_new):
        """ Maps a value from range x \in [x_min, x_max] to y \in [y_min, y_max]

        Arguments
        ---------
        value: The value to be mapped
        range_old: The range the value alreade belongs
        range_new: The new range to map the value
        """
        assert range_old[1] > range_old[0]
        assert range_new[1] > range_new[0]
        return (((value - range_old[0]) * (range_new[1] - range_new[0])) / (range_old[1] - range_old[0])) + range_new[0]

    def send_applied_wrench(self, wrench, actuator_name):
        for actuator in self.model.actuator_names:
            for i in range(0, 6):
                name = actuator_name + '_' + str(i) + '_wrench_actuator'
                if actuator == name:
                    self.sim.data.ctrl[self.model.actuator_name2id(actuator)] = wrench[i]
