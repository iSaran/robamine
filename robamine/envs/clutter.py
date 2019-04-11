#!/usr/bin/env python3
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import os

import robamine.utils as arl
import math

import cv2
from mujoco_py.cymj import MjRenderContext



class PDController:
    """
    Implements a PDController

    Parameters
    ----------
    mass : float
        The mass of the object to control
    damping_ratio : float
        The damping ratio. Defaults to 1 (critically damped system)
    step_response : float
        The response time. Defaults to 0.01 seconds.
    """
    def __init__(self, mass, damping_ratio = 1, step_response = 0.01):
        natural_frequency = 4.0 / step_response
        self.stiffness = mass * natural_frequency * natural_frequency
        self.damping = mass * 2.0 * damping_ratio * natural_frequency

    def get_control(self, pos_error, vel_error):
        return self.stiffness * pos_error + self.damping * vel_error


class Push:
    """
    Defines a push primitive action as defined in kiatos19, with the difference
    of instead of using 4 discrete directions, now we have a continuous angle
    (direction_theta) from which we calculate the direction.
    """
    def __init__(self, initial_pos = np.array([0, 0]), distance = 0.2, direction_theta = 0.0, target = True, object_height = 0.06, z_offset=0.02):
        self.initial_pos, self.distance, self.direction_theta, self.target, self.object_height, self.z_offset = initial_pos, distance, direction_theta, target, object_height, z_offset
        if target:
            self.z = object_height - z_offset
        else:
            self.z = object_height + z_offset
        self.direction = np.array([math.cos(self.direction_theta), math.sin(self.direction_theta)])

    def __str__(self):
        return "Initial Position: " + str(self.initial_pos) + "\n" + \
               "Distance: " + str(self.distance) + "\n" + \
               "Direction: " + str(self.direction) + "\n" + \
               "z: " + str(self.z) + "\n"


class Clutter(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__),
                            "assets/xml/robots/clutter.xml")

        self.model = load_model_from_path(path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self._viewers = {}

        self.ctx = MjRenderContext(self.sim)

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self.action_space = spaces.Box(low=np.array([0]),
                                       high=np.array([2 * math.pi]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([-1, -1, -1]),
                                            high=np.array([1, 1, 1]),
                                            dtype=np.float32)

        self.object_names = ['object1', 'object2', 'object3']
        self.pd = PDController(mass = arl.mujoco.get_body_mass(self.sim.model, 'finger'))

        # Initialize this parent class because our environment wraps Mujoco's  C/C++ code.
        utils.EzPickle.__init__(self)
        self.seed()

    def reset_model(self):
        # Randomize the position of the obstracting objects
        random_qpos = self.init_qpos
        for object_name in self.object_names:
            index = self.sim.model.get_joint_qpos_addr(object_name)
            r = abs(np.random.normal(0, 0.01)) + 0.05
            theta = np.random.uniform(0, 2*math.pi)
            random_qpos[index[0]] = r * math.cos(theta)
            random_qpos[index[0]+1] = r * math.sin(theta)

        self.set_state(random_qpos, self.init_qvel)
        return self.get_obs()

    def get_obs(self):
        # TODO: Read depth and extract height map as observation. Now return
        # random observation. Also change in the constructor the observation
        # space shape.

        # camera_intrinsics = [525, 525, 1920 / 2, 1080 / 2]
        # point_cloud = arl.depth_to_point_cloud(depth, camera_intrinsics)
        # target_pose = arl.get_body_pose(self.sim, 'target')
        # t = self.sim.data.get_camera_xpos('xtion')

        rgb, depth = self.sim.render(1920, 1080, depth=True, camera_name='xtion', mode='offscreen')
        cv2.imwrite("/home/mkiatos/Desktop/obs.png", rgb)
        return rgb
        # return self.observation_space.sample()

    def step(self, action):
        done = False
        obs = self.get_obs()
        reward = self.get_reward(obs)
        time = self.do_simulation(action)
        if self.terminal_state(obs):
            done = True
        return obs, reward, done, {}

    def do_simulation(self, action):
        time = self.sim.data.time

        push = Push(direction_theta=action[0])

        self.move_joint_to_target('finger', [push.initial_pos[0], push.initial_pos[1], None])
        self.move_joint_to_target('finger', [None, None, push.z])
        end = push.initial_pos + push.distance * push.direction
        self.move_joint_to_target('finger', [end[0], end[1], None])
        self.move_joint_to_target('finger', [None, None, 0.2])
        self.move_joint_to_target('finger', [0, 0, 0.2])

        return time

    def viewer_setup(self):
        # Set the camera configuration (spherical coordinates)
        self.viewer.cam.distance = 0.75
        self.viewer.cam.elevation = -90  # default -90
        self.viewer.cam.azimuth = 90

    def get_reward(self, observation):
        # TODO: Define reward based on observation
        reward = 0
        return reward

    def terminal_state(self, observation):
        return False

    def move_joint_to_target(self, joint_name, target_position, duration = 2):
        """
        Generates a trajectory in Cartesian space (x, y, z) from the current
        position of a joint to a target position. If one of the x, y, z is None
        then the joint will not move in this direction. For example:
        target_position = [None, 1, 1] will move along a trajectory in y,z and
        x will remain the same.

        TODO: The indexes of the actuators are hardcoded right now assuming
        that 0-6 is the actuator of the given joint
        """
        current_pos = self.sim.data.get_joint_qpos(joint_name)
        current_vel = self.sim.data.get_joint_qvel(joint_name)
        current_time = self.sim.data.time
        init_time = self.sim.data.time

        if target_position[0] is not None:
            trajectory_x = arl.Trajectory([current_time, current_time + duration], [current_pos[0], target_position[0]])
        else:
            trajectory_x = arl.Trajectory([current_time, current_time + duration], [current_pos[0], current_pos[0]])
        if target_position[1] is not None:
            trajectory_y = arl.Trajectory([current_time, current_time + duration], [current_pos[1], target_position[1]])
        else:
            trajectory_y = arl.Trajectory([current_time, current_time + duration], [current_pos[1], current_pos[1]])
        if target_position[2] is not None:
            trajectory_z = arl.Trajectory([current_time, current_time + duration], [current_pos[2], target_position[2]])
        else:
            trajectory_z = arl.Trajectory([current_time, current_time + duration], [current_pos[2], current_pos[2]])

        while current_time <= init_time + duration:
            # TODO: The indexes of the actuators are hardcoded right now
            # assuming that 0-6 is the actuator of the given joint
            self.sim.data.ctrl[0] = self.pd.get_control(trajectory_x.pos(current_time) - current_pos[0], trajectory_x.vel(current_time) - current_vel[0])
            self.sim.data.ctrl[1] = self.pd.get_control(trajectory_y.pos(current_time) - current_pos[1], trajectory_y.vel(current_time) - current_vel[1])
            self.sim.data.ctrl[2] = self.pd.get_control(trajectory_z.pos(current_time) - current_pos[2], trajectory_z.vel(current_time) - current_vel[2])
            self.sim.data.ctrl[3] = 0.0
            self.sim.data.ctrl[4] = 0.0
            self.sim.data.ctrl[5] = 0.0

            self.sim.step()
            self.render()

            current_pos = self.sim.data.get_joint_qpos(joint_name)
            current_vel = self.sim.data.get_joint_qvel(joint_name)
            current_time = self.sim.data.time

