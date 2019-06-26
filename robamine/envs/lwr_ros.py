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
from std_srvs.srv import Trigger

from time import sleep

ACTION_TOPIC_NAME = 'impedance_actions'
OBS_TOPIC_NAME = 'force_observations'

class LWRROS(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=np.array([0, - math.pi]),
                                       high=np.array([0.1, math.pi]),
                                       dtype=np.float32)

        # 300 size: 100 samples of 3d forces
        self.observation_space = spaces.Box(low=np.full((300,), -1),
                                            high=np.full((300,), 1),
                                            dtype=np.float32)

        rospy.init_node('lwr_ros_env')
        self.pub = rospy.Publisher(ACTION_TOPIC_NAME, ImpedanceActions, queue_size=10)
        rospy.Subscriber(OBS_TOPIC_NAME, ForceObservations, self.sub_callback)
        terminal_state_srv = rospy.Service('/terminal_state', Trigger, self.terminal_state_cb)
        self.is_terminal_state = False;
        self.rate = rospy.Rate(100)
        self.data = np.zeros((300))
        self.reward_data = 0.0
        self.mutex = Lock()
        self.first_msg_arrived = False

    def terminal_state_cb(self, req):
        self.is_terminal_state = True
        return [True, '']

    def sub_callback(self, data):

        self.first_msg_arrived = True
        local = np.zeros((len(self.data)))

        for i in range(len(data.obs)):
            local[3 * i + 0] = data.obs[i].x
            local[3 * i + 1] = data.obs[i].y
            local[3 * i + 2] = data.obs[i].z

        self.mutex.acquire()
        self.data = local
        self.reward_data = data.reward
        self.mutex.release()

    def reset(self):
        print('Resetting env')
        print('Waiting for service "/reset"')
        rospy.wait_for_service('/reset')
        try:
            srv = rospy.ServiceProxy('/reset', Trigger)
            resp1 = srv()
            print('Called service /reset successfully')
        except (rospy.ServiceException, e):
            print("Service call failed: %s"%e)
            time.sleep(button_delay)
        return np.zeros(300)

    def render(self):
        return

    def get_obs(self):
        self.mutex.acquire()
        local = self.data
        self.mutex.release()
        return local

    def step(self, action):

        assert not np.any(np.isnan(action)), "LWRROS env: at least one action is NaN."

        while not rospy.is_shutdown() and not self.first_msg_arrived:
            rospy.logwarn_throttle(10, "LWRROS env: Waiting for the first msg in " + OBS_TOPIC_NAME + " to arrive.")
            sleep(0.01)

        self.rate.sleep()
        done = False
        time = self.send_command(action)
        obs = self.get_obs()

        assert not np.any(np.isnan(obs)), "LWRROS env: at least one observation is NaN."

        reward = self.get_reward(obs)
        if self.terminal_state(obs):
            done = True
        return obs, reward, done, {}

    def send_command(self, action):
        a = ImpedanceActions()


        # Agent gives actions between [-1, 1]. Convert to the action ranges of
        # the action space of the environment
        my_action = action.copy()
        agent_high = 1
        agent_low = -1
        env_low = [0, -math.pi]
        env_high = [0.2, math.pi, math.pi]
        for i in range(len(my_action)):
            my_action[i] = (((my_action[i] - agent_low) * (env_high[i] - env_low[i])) / (agent_high - agent_low)) + env_low[i]

        a.amplitude = my_action[0]
        a.angle = my_action[1]
        if not rospy.is_shutdown():
            time = rospy.get_time()
            self.pub.publish(a)
        return time

    def get_reward(self, observation):
        return self.reward_data

    def terminal_state(self, observation):
        if self.is_terminal_state:
            self.is_terminal_state = False
            return True
        return False

