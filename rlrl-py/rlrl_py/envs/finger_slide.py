#!/usr/bin/env python3
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer

from mujoco_py import functions
from mujoco_py.generated import const

from gym import utils, spaces
from gym.envs.robotics import robot_env
import os

import rlrl_py.utils as arl
import random

class FingerSlide(robot_env.RobotEnv, utils.EzPickle):
    def __init__(self):
        # Create MuJoCo Model
        path = os.path.join(os.path.dirname(__file__),
                            "assets/xml/robots/small_table_pillbox_finger.xml")
        self.model = load_model_from_path(path)
        self.first_time = True

        self.n_actions = 1
        init_qpos = {}
        robot_env.RobotEnv.__init__(self, path, init_qpos, n_actions=self.n_actions, n_substeps=1)
        utils.EzPickle.__init__(self)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, desired_goal, info):
        return 0.0

    # RobotEnv methods
    # ----------------------------

    def _is_success(self, achieved_goal, desired_goal):
        return False

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        action = self.map_to_new_range(action, [-1, 1], [-1, 0])

        # Calculate the bias, i.e. the sum of coriolis, centrufugal, gravitational force
        # Command the joints to the initial joint configuration in order to
        # have the joints fixed in this position (no falling or external forces
        # are affecting the fingers) and also command the actions as forces on
        # the wrist
        addr = self.model.get_joint_qvel_addr("finger_free_joint")
        bias = self.sim.data.qfrc_bias[addr[0]:addr[1]]
        commanded = [0.0, 0.0, action, 0.0, 0.0, 0.0]
        #commanded[0:3] = np.matmul(self.sim.data.get_body_xmat('finger'), commanded[0:3])
        applied_force = bias + commanded
        self.send_applied_wrench(applied_force, 'finger')

        t = self.sim.data.time
        if t > 1:
            T = arl.trajectory.Trajectory([1, 1.125], [0, 1])
            T2 = arl.trajectory.Trajectory([1.125, 1], [1, 0])
            if t < 1.5:
                dist = T.pos(t)
            else:
                dist = T2.pos(t)
            disturbance = [dist, 0.0, 0, 0.00, 0.0, 0.0]
            self.send_applied_wrench(disturbance, 'pillbox')

        #print(self.get_contact_force("optoforce", "pillbox"))

       ######################################################## #########33functions.mj_inverse(self.model, self.sim.data)
       ######################################################## #########33ids = self.model.get_joint_qvel_addr("finger_free_joint")
       ######################################################## #########33print(self.sim.data.qfrc_inverse[ids[0]:ids[1]])
       ######################################################## #########33print(world_wrench)
       ######################################################## #########33world_wrench = self.sim.data.qfrc_inverse[ids[0]:ids[1]]
       ######################################################## # wrist_pos = np.subtract(self.sim.data.get_body_xpos('pillbox'), self.sim.data.get_body_xpos('bh_wrist'))
       ######################################################## # wrist_rot = np.matmul(np.transpose(self.sim.data.get_body_xmat('bh_wrist')), self.initial_obj_rot_mat)
       ######################################################## # screw = arl.orientation.screw_transformation(wrist_pos, wrist_rot)
       ######################################################## # force_wrist = np.matmul(screw, force_object)
       ######################################################## # force_wrist_world = np.matmul(arl.orientation.rotation_6x6(self.sim.data.get_body_xmat('bh_wrist')), force_wrist)

    def _get_obs(self):
        return {
            'observation': np.zeros(shape=(3, 1)),
            'achieved_goal': np.zeros(shape=(3, 1)),
            'desired_goal': np.zeros(shape=(3, 1))
            }

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('pillbox')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        # Set the camera configuration (spherical coordinates)
        self.viewer.cam.distance = 0.846
        self.viewer.cam.elevation = -21.774
        self.viewer.cam.azimuth = -148.306

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.initial_obj_rot_mat = self.sim.data.get_body_xmat('pillbox')

        self.disturbance = np.random.normal(2, 1, 2)
        self.disturbance[0] = random.choice([-1, 1]) * self.disturbance[0]
        self.disturbance = [0, 0]
        return True

    def _render_callback(self):
        if self.first_time:
            self.init_goal = self.goal
            self.first_time = False

    def terminal_state(self, observation):
        if np.linalg.norm(observation[0:3]) > 0.5:
            return True

        return False

    def _sample_goal(self):
        "Samples a goal for the object w.r.t the frame of the object."
        mean = 0.035
        dev = 0.01
        sample_y = - np.random.normal(mean, dev, 1)
        sampled_goal = np.array([0, sample_y[0], 0, 0, sample_y[0] - 0.015, 0.02])
        return sampled_goal

    def get_contact_force(self, geom1_name, geom2_name):
        result = np.zeros(shape=6)
        have_contact = False
        for i in range(0, self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if (self.model.geom_id2name(contact.geom1) == geom1_name) and (self.model.geom_id2name(contact.geom2) == geom2_name):
                functions.mj_contactForce(self.model, self.sim.data, i, result)
                have_contact = True
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
