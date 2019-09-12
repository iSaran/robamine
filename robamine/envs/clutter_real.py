"""
Clutter
=======


This module contains the implementation of a cluttered environment, based on
:cite:`kiatos19`.
"""
import gym
from gym import spaces
import numpy as np

import rospy
from rosba_msgs.srv import Push

default_params = {
        'name' : 'ClutterReal-v0',
        'discrete' : True,
        'nr_of_actions' : 16,
        'nr_of_obstacles' : [5, 10],
        'push_distance' : 0.1,
        'target_height_range' : [.005, .01],
        'obstacle_height_range' : [.005, .02],
        'split' : True
        }

class ClutterReal():
    def __init__(self, params = default_params):
        self.params = params

        if self.params['discrete']:
            self.action_space = spaces.Discrete(self.params['nr_of_actions'])
        else:
            self.action_space = spaces.Box(low=np.array([-1, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)

        if self.params['split']:
            obs_dim = int(self.params['nr_of_actions'] / 2) * 263
        else:
            obs_dim = 263

        self.observation_space = spaces.Box(low=np.full((obs_dim,), 0),
                                            high=np.full((obs_dim,), 0.3),
                                            dtype=np.float32)

        rospy.init_node('clutter_real_env')

        # Services
        serf.push_srv = rospy.Service('/push', Push, handle_add_two_ints)

    def reset_model(self):
        observation, _, _ = self.get_obs()

        self.last_timestamp = self.sim.data.time
        self.success = False
        return observation

    def get_obs(self):
        # Call feature extraction service
        return final_feature, points_above_table, bbox

    def step(self, action):
        done = False
        time = self._push(action)
        experience_time = time - self.last_timestamp
        self.last_timestamp = time
        obs, pcd, dim = self.get_obs()
        reward = self.get_reward(obs, pcd, dim)
        if self.terminal_state(obs):
            done = True
        return obs, reward, done, {'experience_time': experience_time, 'success': self.success}

    def _push(self, action):
        rospy.wait_for_service('push')
        try:
            pushing = rospy.ServiceProxy('push', Push)
            response = pushing(action, self.params['push_distance'])
            return resp1.sum
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def do_simulation(self, action):
        if self.params['discrete']:
            myaction = action
            push_target = True
            if myaction > int(self.params['nr_of_actions'] / 2) - 1:
                push_target = False
                myaction -= int(self.params['nr_of_actions'] / 2)
            theta = myaction * 2 * math.pi / (self.action_space.n / 2)
            push = Push(direction_theta=theta, distance=self.params['push_distance'], object_height = self.target_height, target=push_target, object_length = self.target_length, object_width = self.target_width, finger_size = self.finger_length)
        else:
            my_action = action.copy()

            # agent gives actions between [-1, 1]. convert to the action ranges of
            # the action space of the environment
            agent_high = 1
            agent_low = -1
            env_low = [-math.pi, 0]
            env_high = [math.pi, 1]
            for i in range(len(my_action)):
                my_action[i] = (((my_action[i] - agent_low) * (env_high[i] - env_low[i])) / (agent_high - agent_low)) + env_low[i]

            if my_action[1] > 0.5:
                push_target = True
            else:
                push_target = False

            push = Push(direction_theta=my_action[0], distance=self.params['push_distance'], object_height = self.target_height, target=push_target, object_length = self.target_length, object_width = self.target_width, finger_size = self.finger_length)

        # Transform pushing from target frame to world frame
        push_direction = np.array([push.direction[0], push.direction[1], 0])
        push_direction_world = np.matmul(self.target_quat.rotation_matrix(), push_direction)
        push_initial_pos = np.array([push.initial_pos[0], push.initial_pos[1], 0])
        push_initial_pos_world = np.matmul(self.target_quat.rotation_matrix(), push_initial_pos) + self.target_pos

        init_z = 2 * self.target_height + 0.05
        self.sim.data.set_joint_qpos('finger', [push_initial_pos_world[0], push_initial_pos_world[1], init_z, 1, 0, 0, 0])
        self.sim_step()
        if self.move_joint_to_target('finger', [None, None, push.z], stop_external_forces=True):
            end = push_initial_pos_world[:2] + push.distance * push_direction_world[:2]
            self.move_joint_to_target('finger', [end[0], end[1], None])
        else:
            self.push_stopped_ext_forces = True

        return self.sim.data.time

    def viewer_setup(self):
        # Set the camera configuration (spherical coordinates)
        self.viewer.cam.distance = 0.75
        self.viewer.cam.elevation = -90  # default -90
        self.viewer.cam.azimuth = 90

    def get_reward(self, observation, point_cloud, dim):
        reward = 0.0

        # Penalize external forces during going downwards
        if self.push_stopped_ext_forces:
            return -10

        if min([observation[-4], observation[-3], observation[-2], observation[-1]]) < 0:
            return -10

        # for each push that frees the space around the target
        points_around = []
        gap = 0.03
        bbox_limit = 0.01
        for p in point_cloud:
            if (-dim[0] - bbox_limit > p[0] > -dim[0] - gap - bbox_limit or \
             dim[0] + bbox_limit < p[0] < dim[0] + gap + bbox_limit) and \
             -dim[1]  < p[1] < dim[1]:
                points_around.append(p)
            if (-dim[1] - bbox_limit > p[1] > -dim[1] - gap - bbox_limit or \
            dim[1] + bbox_limit < p[1] < dim[1] + gap + bbox_limit) and \
            -dim[0]  < p[0] < dim[0]:
                points_around.append(p)

        if self.no_of_prev_points_around == len(points_around):
            return -5

        self.no_of_prev_points_around = len(points_around)

        if len(points_around) == 0:
            return +10

        max_cost = -5

        return -1

        # k = max(self.no_of_prev_points_around, len(points_around))
        # if k != 0:
        #     reward = (self.no_of_prev_points_around - len(points_around)) / k
        # else:
        #     reward = 0.0
        # reward *= 10.0
        # self.no_of_prev_points_around = len(points_around)

        # cv_tools.plot_point_cloud(point_cloud)
        # cv_tools.plot_point_cloud(points_around)

        # Penalize the agent as it gets the target object closer to the edge
        # max_cost = -5
        # reward += sigmoid(observation[-1], a=max_cost, b=-15/max(self.surface_size), c=-4)
        # if observation[-1] < 0:
        #     reward = -10

        # For each object push
        # reward += -1
        # return reward

    def terminal_state(self, observation):

        # Terminal if collision is detected
        if self.push_stopped_ext_forces:
            self.push_stopped_ext_forces = False
            return True

        # Terminate if the target flips to its side, i.e. if target's z axis is
        # parallel to table, terminate.
        target_z = self.target_quat.rotation_matrix()[:,2]
        world_z = np.array([0, 0, 1])
        if abs(np.dot(target_z, world_z)) < 0.1:
            return True

        # If the object has fallen from the table
        if min([observation[-4], observation[-3], observation[-2], observation[-1]]) < 0:
            return True

        # If the object is free from obstacles around (no points around)
        if self.no_of_prev_points_around == 0:
            self.success = True
            return True

        return False
