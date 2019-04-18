"""
LWR ROS
=======

Gym environment for communicating with LWR robot through ROS.
"""
#!/usr/bin/env python
import gym
from gym import spaces
import numpy as np
import math

import rospy
from std_msgs.msg import String

ACTION_TOPIC_NAME = 'lwr_ros_action'
OBS_TOPIC_NAME = 'lwr_ros_action'

class LWRROS(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=np.array([0]),
                                       high=np.array([2 * math.pi]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([-1, -1, -1]),
                                            high=np.array([1, 1, 1]),
                                            dtype=np.float32)

        rospy.init_node('lwr_ros_env')
        self.pub = rospy.Publisher(ACTION_TOPIC_NAME, String, queue_size=10)
        rospy.Subscriber(OBS_TOPIC_NAME, String, self.sub_callback)
        self.rate = rospy.Rate(100)
        self.data = None

    def sub_callback(self, data):
        self.data = data.data
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def reset(self):
        return

    def render(self):
        return

    def get_obs(self):
        return self.observation_space.sample()

    def step(self, action):
        self.rate.sleep()
        done = False
        obs = self.get_obs()
        reward = self.get_reward(obs)
        time = self.send_command(action)
        if self.terminal_state(obs):
            done = True
        return obs, reward, done, {}

    def send_command(self, action):
        if not rospy.is_shutdown():
            time = rospy.get_time()
            self.pub.publish('fdsa')
        return time

    def get_reward(self, observation):
        reward = 0
        return reward

    def terminal_state(self, observation):
        return False

