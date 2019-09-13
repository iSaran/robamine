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
from rosba_msgs.srv import Push, ExtractFeatures
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
        self.success = False
        rospy.init_node('clutter_real_env')

    def reset(self):
        # self._go_home()
        logger.warn('New episode :)')
        input("Resetting env. Set up the objects on the table and press enter to continue...")
        self._detect_target()
        observation = self.get_obs()
        print(observation)
        self.success = False
        return observation

    def get_obs(self):
        # Call feature extraction service
        logger.info('Calling feature extraction service...')

        rospy.wait_for_service('extract_features')
        try:
            extract_feature = rospy.ServiceProxy('extract_features', ExtractFeatures)
            response = extract_feature()
            logger.info('Feature received')
            return response
        except rospy.ServiceException:
            logger.error('Extract feature service failed')

        logger.info('Feature received')
        return np.zeros(263), np.zeros(500), 1

    def step(self, action):
        done = False
        success = self._push(action)
        if not success:
            self.push_failed = True
        experience_time = 0
        self._go_home()
        self._detect_target()
        obs = self.get_obs()
        # reward = self.get_reward(obs, pcd, dim)
        reward = -1;
        if self.terminal_state(obs):
            done = True
        return obs, reward, done, {'experience_time': experience_time, 'success': self.success}

    def _detect_target(self):
        logger.info('Calling detect target service...')
        rospy.wait_for_service('estimate_pose')
        try:
            pushing = rospy.ServiceProxy('estimate_pose', Trigger)
            response = pushing()
            return response
        except rospy.ServiceException:
            logger.error('estimate_pose service failed')

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
        return float(input('Enter reward: '))

    def terminal_state(self, observation):
        k = int(input('Terminal state?: '))
        if k > 0:
            l = int(input('Was the episode successful?: '))
            if l > 0:
                self.success = True
            else:
                self.success = False
            return True
        return False
