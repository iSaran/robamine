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
from std_srvs.srv import Trigger

import logging
logger = logging.getLogger('robamine.env.clutter_real')

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

class ClutterReal(gym.Env):
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

        self.push_failed = False
        rospy.init_node('clutter_real_env')

    def reset(self):
        self._go_home()
        input("Resetting env. Set up the objects on the table and press enter to continue...")
        observation = self.get_obs()
        return observation

    def get_obs(self):
        # Call feature extraction service
        logger.info('Calling feature extraction service...')

        # rospy.wait_for_service('extract_feature')
        # try:
        #     extract_feature = rospy.ServiceProxy('extract_feature', Push)
        #     response = pushing()
        #     logger.info('Feature received')
        #     return response
        # except rospy.ServiceException:
        #     logger.error('Extract feature service failed')

        # logger.info('Feature received')
        return np.zeros(263), np.zeros(500), 1



    def step(self, action):
        done = False
        success = self._push(action)
        if not success:
            self.push_failed = True
        experience_time = 0
        self._go_home()
        self._detect_target()
        obs, pcd, dim = self.get_obs()
        reward = self.get_reward(obs, pcd, dim)
        if self.terminal_state(obs):
            done = True
        return obs, reward, done, {'experience_time': experience_time, 'success': self.success}

    def _detect_target(self):
        logger.info('Calling detect target service...')
        rospy.wait_for_service('detect_target')
        try:
            pushing = rospy.ServiceProxy('go_home', Push)
            response = pushing()
            return response
        except rospy.ServiceException:
            logger.error('Go home service failed')

    def _go_home(self):
        logger.info('Calling go home service...')
        rospy.wait_for_service('go_home')
        try:
            pushing = rospy.ServiceProxy('go_home', Trigger)
            response = pushing()
            return response
        except rospy.ServiceException:
            logger.error('Go home service failed')

    def _push(self, action):
        logger.info('Calling pushing service...')
        rospy.wait_for_service('push')
        try:
            pushing = rospy.ServiceProxy('push', Push)
            response = pushing(action, self.params['push_distance'], self.params['nr_of_actions'], self.params['extra_primitive'])
            return response
        except rospy.ServiceException:
            logger.error('Push service failed')

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

    def terminal_state(self, observation):

        # Terminal if collision is detected
        if self.push_failed:
            self.push_failed = False
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
