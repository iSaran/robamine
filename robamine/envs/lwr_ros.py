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
from std_msgs.msg import Float64
from fog_msgs.msg import ForceObservations, ImpedanceActions
from threading import Lock

ACTION_TOPIC_NAME = 'impedance_actions'
OBS_TOPIC_NAME = 'force_observations'

class LWRROS(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=np.array([0, - math.pi]),
                                       high=np.array([0.1, math.pi]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([-1, -1, -1]),
                                            high=np.array([1, 1, 1]),
                                            dtype=np.float32)

        rospy.init_node('lwr_ros_env')
        self.pub = rospy.Publisher(ACTION_TOPIC_NAME, ImpedanceActions, queue_size=10)
        rospy.Subscriber(OBS_TOPIC_NAME, ForceObservations, self.sub_callback)
        self.rate = rospy.Rate(100)
        self.data = None
        self.reward_data = 0.0
        self.mutex = Lock()

    def sub_callback(self, data):

        local = np.zeros((len(data.obs), 3))

        for i in range(len(data.obs)):
            local[i][0] = data.obs[i].x
            local[i][1] = data.obs[i].y
            local[i][2] = data.obs[i].z

        self.mutex.acquire()
        self.data = local
        self.reward_data = data.reward
        self.mutex.release()

    def reset(self):
        return

    def render(self):
        return

    def get_obs(self):
        self.mutex.acquire()
        local = self.data
        self.mutex.release()
        return local

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
        a = ImpedanceActions()
        a.amplitude = action[0]
        a.angle = action[1]
        if not rospy.is_shutdown():
            time = rospy.get_time()
            self.pub.publish(a)
        return time

    def get_reward(self, observation):
        return self.reward_data

    def terminal_state(self, observation):
        return False

