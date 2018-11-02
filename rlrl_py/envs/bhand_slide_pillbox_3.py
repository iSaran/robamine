#!/usr/bin/env python3
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym import utils, spaces
from gym.envs.robotics import robot_env
import os

import robamine.utils as arl
import random

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class BHandSlidePillbox3(robot_env.RobotEnv, utils.EzPickle):
    def __init__(self, distance_threshold = 0.01, reward_type="sparse", min_action = -5, max_action = 0):
        """ Initializes a new BHand Slide Pillbox environment

        Args:
            distance_threshold (float): the threshold after which the goal is considered achieved
        """

        # Parse arguments
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.min_action = min_action
        self.max_action = max_action

        # Create MuJoCo Model
        path = os.path.join(os.path.dirname(__file__),
                            "assets/xml/robots/small_table_floating_bhand.xml")
        self.model = load_model_from_path(path)

        # TODO(isaran): Do not use hardcoded indeces, use native supported functions
        # for extracting the addresses of the sensors in sensordata
        self.sensor_name = {
            'optoforce_1': [0, 1, 2],
            'optoforce_2': [3, 4, 5]
        }

        # Initialize the Bhand joint configuration to a prespecified config
        self.init_config = {
            "bh_j11_joint": 0.0,
            "bh_j12_joint": 0.8,
            "bh_j22_joint": 0.8,
            "bh_j32_joint": 0.0
        }

        init_qpos = {
            "bh_j11_joint": self.init_config["bh_j11_joint"],
            "bh_j12_joint": self.init_config["bh_j12_joint"],
            "bh_j22_joint": self.init_config["bh_j22_joint"],
            "bh_j32_joint": self.init_config["bh_j32_joint"],
            "bh_j21_joint": self.init_config["bh_j11_joint"],
            "bh_j13_joint": 0.3333 * self.init_config["bh_j12_joint"],
            "bh_j23_joint": 0.3333 * self.init_config["bh_j22_joint"],
            "bh_j33_joint": 0.3333 * self.init_config["bh_j32_joint"],
            "world_to_pillbox": [0.2, 0.92, 0.3725, 1, 0, 0, 0],
        }
        self.initial_obj_pos = init_qpos['world_to_pillbox'][0:3]

        self.first_time = True

        robot_env.RobotEnv.__init__(self, path, init_qpos, n_actions=1, n_substeps=1)
        utils.EzPickle.__init__(self)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, desired_goal, info):
        ag_object, ag_dominant_centroid = self.get_subgoals(achieved_goal)
        dg_object, dg_dominant_centroid = self.get_subgoals(desired_goal)

        distance_object = goal_distance(ag_object, dg_object)
        distance_dominant_centroid = goal_distance(ag_dominant_centroid, dg_dominant_centroid)
        if self.reward_type == 'sparse':
            object_reached = distance_object < self.distance_threshold
            dominant_centroid_reached = distance_dominant_centroid < self.distance_threshold
            both_reached = np.logical_and(object_reached, dominant_centroid_reached)
            reward = -(~both_reached).astype(np.float32)
            return reward
        else:
            return distance_object + distance_dominant_centroid

    # RobotEnv methods
    # ----------------------------

    def _is_success(self, achieved_goal, desired_goal):
        ag_object, ag_dominant_centroid = self.get_subgoals(achieved_goal)
        dg_object, dg_dominant_centroid = self.get_subgoals(desired_goal)

        distance_object = goal_distance(ag_object, dg_object)
        distance_dominant_centroid = goal_distance(ag_dominant_centroid, dg_dominant_centroid)
        return (distance_object < self.distance_threshold and distance_dominant_centroid < self.distance_threshold).astype(np.float32)

    def _set_action(self, action):
        assert action.shape == (1,)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        # Set the bias to the applied generilized force in order to
        # implement a kind of gravity compensation in bhand.
        for i in range(*self.model.get_joint_qvel_addr("bh_wrist_joint")):
            bias = self.sim.data.qfrc_bias[i]
            self.sim.data.qfrc_applied[i] = bias

        if (self.sim.data.time > 1):
            print('-----')
            x = self.disturbance[0]
            y = self.disturbance[1]
            print(x)
            print(y)
            print('-----')
        else:
            x = 0
            y = 0
        z = self.map_to_new_range(action, (-1, 1), (self.min_action, self.max_action))
        force_object = np.array([x, y, z, 0, 0, 0])

        # Trasfer the desired action (force on the object frame to the wrist and command the wrist
        wrist_pos = np.subtract(self.sim.data.get_body_xpos('pillbox'), self.sim.data.get_body_xpos('bh_wrist'))
        wrist_rot = np.matmul(np.transpose(self.sim.data.get_body_xmat('bh_wrist')), self.initial_obj_rot_mat)
        screw = arl.orientation.screw_transformation(wrist_pos, wrist_rot)
        force_wrist = np.matmul(screw, force_object)
        print(force_wrist)
        force_wrist_world = np.matmul(arl.orientation.rotation_6x6(self.sim.data.get_body_xmat('bh_wrist')), force_wrist)

        # Command the joints to the initial joint configuration in order to
        # have the joints fixed in this position (no falling or external forces
        # are affecting the fingers) and also command the actions as forces on
        # the wrist
        for actuator in self.model.actuator_names:
            if actuator.startswith('bh_j'):
                self.sim.data.ctrl[self.model.actuator_name2id(actuator)] = self.init_config[actuator]
            if actuator == 'bh_wrist_force_x_actuator':
                self.sim.data.ctrl[self.model.actuator_name2id(actuator)] = force_wrist_world[0]
            if actuator == 'bh_wrist_force_y_actuator':
                self.sim.data.ctrl[self.model.actuator_name2id(actuator)] = force_wrist_world[1]
            if actuator == 'bh_wrist_force_z_actuator':
                self.sim.data.ctrl[self.model.actuator_name2id(actuator)] = force_wrist_world[2]
            if actuator == 'bh_wrist_torque_x_actuator':
                self.sim.data.ctrl[self.model.actuator_name2id(actuator)] = force_wrist_world[3]
            if actuator == 'bh_wrist_torque_y_actuator':
                self.sim.data.ctrl[self.model.actuator_name2id(actuator)] = force_wrist_world[4]
            if actuator == 'bh_wrist_torque_z_actuator':
                self.sim.data.ctrl[self.model.actuator_name2id(actuator)] = force_wrist_world[5]


    def _get_obs(self):
        # Calculate the dominant point, i.e. the centroid of the dominant fingers
        centroid = (self.sim.data.get_body_xpos('wam/bhand/finger_1/tip_link') + self.sim.data.get_body_xpos('wam/bhand/finger_2/tip_link')) / 2
        dominant_point = centroid - self.initial_obj_pos

        # Calculate the norms of the optoforce readings
        optoforce1 = np.linalg.norm(self.sim.data.sensordata[self.sensor_name["optoforce_1"]])
        optoforce2 = np.linalg.norm(self.sim.data.sensordata[self.sensor_name["optoforce_2"]])

        # Read the object's position
        obj_position = self.sim.data.get_body_xpos('pillbox') - self.initial_obj_pos

        # Concatenate all the above to the observation vector
        obs = np.concatenate((obj_position.copy(), dominant_point, [optoforce1], [optoforce2]))
        achieved_goal = np.concatenate((np.squeeze(obj_position.copy()), np.squeeze(dominant_point.copy())))

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
            }

        return obs_dict

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
        self.viewer.cam.distance = 0.559478
        self.viewer.cam.elevation = -21.928236
        self.viewer.cam.azimuth = 9.089018

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
        index = self.sim.model.get_joint_qpos_addr("world_to_pillbox")
        initial_world_to_pillbox_pos = self.initial_state.qpos[index[0]:index[0]+3]
        world_to_pillbox_pos = self.sim.data.get_joint_qpos("world_to_pillbox")[0:3]
        offset = world_to_pillbox_pos - initial_world_to_pillbox_pos
        site_id = self.sim.model.site_name2id('target')
        self.sim.model.site_pos[site_id] = self.goal[0:3] - offset

        centroid_site_id = self.sim.model.site_name2id('dominant_centroid')
        self.sim.model.site_pos[centroid_site_id] = (self.sim.data.get_body_xpos('wam/bhand/finger_1/tip_link') + self.sim.data.get_body_xpos('wam/bhand/finger_2/tip_link')) / 2
        self.sim.forward()

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

    def get_initial_state(self, model_path):
        # Load local copy of the Mujoco models to have access to joint names and ids
        if model_path.startswith('/'):
            path = model_path
        else:
            path = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(path):
            raise IOError('File {} does not exist'.format(path))

        model = load_model_from_path(path)
        sim = MjSim(model)
        viewer = None
        sim.forward()

        init_qpos = sim.data.qpos
        for key in self.init_config:
            init_qpos[sim.model.get_joint_qpos_addr(key)]  = self.init_config[key];

        object_pos = sim.data.get_body_xpos('pillbox')

        return init_qpos, object_pos

    # Auxialiary methods
    # ----------------------------
    def get_subgoals(self, goal):
        if goal.ndim == 1:
            subgoal_1 = goal[0:3]
            subgoal_2 = goal[3:6]
        else:
            subgoal_1 = goal[:, 0:3]
            subgoal_2 = goal[:, 3:6]
        return subgoal_1, subgoal_2

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
