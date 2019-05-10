"""
Clutter
=======


This module contains the implementation of a cluttered environment, based on
:cite:`kiatos19`.
"""
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import os

from robamine.utils.robotics import PDController, Trajectory
from robamine.utils.mujoco import get_body_mass, get_body_pose, get_camera_pose, get_geom_size, get_body_inertia
from robamine.utils.orientation import Quaternion
import robamine.utils.cv_tools as cv_tools
import math

from robamine.utils.orientation import rot2quat

import cv2
from mujoco_py.cymj import MjRenderContext

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
        self._viewers = {}
        self.offscreen = MjRenderContextOffscreen(self.sim, 0)
        self.viewer = MjViewer(self.sim)

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self.action_space = spaces.Box(low=np.array([0]),
                                       high=np.array([2 * math.pi]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([-1, -1, -1]),
                                            high=np.array([1, 1, 1]),
                                            dtype=np.float32)

        self.object_names = ['object1', 'object2', 'object3']

        finger_mass = get_body_mass(self.sim.model, 'finger')
        self.pd = PDController.from_mass(mass = finger_mass, step_response=0.005)

        moment_of_inertia = get_body_inertia(self.sim.model, 'finger')
        self.pd_rot = []
        self.pd_rot.append(PDController.from_mass(mass = moment_of_inertia[0], step_response=0.005))
        self.pd_rot.append(PDController.from_mass(mass = moment_of_inertia[1], step_response=0.005))
        self.pd_rot.append(PDController.from_mass(mass = moment_of_inertia[2], step_response=0.005))

        # Parameters, updated once during reset of the model
        # State variables. Updated after each call in self.sim_step()
        self.time = 0.0
        self.finger_position = np.zeros(3)
        self.finger_quat = Quaternion()
        self.finger_quat_prev = Quaternion()
        self.finger_vel = np.zeros(6)
        self.target_height = 0.0
        self.target_length = 0.0
        self.target_width = 0.0

        # Initialize this parent class because our environment wraps Mujoco's  C/C++ code.
        utils.EzPickle.__init__(self)
        self.seed()

    def reset_model(self):
        target_size = get_geom_size(self.sim.model, 'target')
        # Randomize the position of the obstracting objects
        random_qpos = self.init_qpos
        for object_name in self.object_names:
            index = self.sim.model.get_joint_qpos_addr(object_name)
            r = abs(np.random.normal(0, 0.01)) + 3 * max([target_size[0], target_size[1]])
            theta = np.random.uniform(0, 2*math.pi)
            random_qpos[index[0]] = r * math.cos(theta)
            random_qpos[index[0]+1] = r * math.sin(theta)

        # Set the initial position of the finger outside of the table, in order
        # to not occlude the objects during reading observation from the camera
        index = self.sim.model.get_joint_qpos_addr('finger')
        table_size = get_geom_size(self.sim.model, 'table')
        table_pos = self.sim.data.get_body_xpos('table')
        init_finger_pos = table_pos + 1.1 * table_size
        random_qpos[index[0]]   = init_finger_pos[0]
        random_qpos[index[0]+1] = init_finger_pos[1]
        random_qpos[index[0]+2] = init_finger_pos[2]

        self.set_state(random_qpos, self.init_qvel)

        # Move forward the simulation to be sure that the objects have landed
        for _ in range(200):
            self.sim_step()

        return self.get_obs()

    def get_obs(self):
        """
        Read depth and extract height map as observation
        :return:
        """
        self._move_finger_outside_the_table()

        # Get the depth image
        self.offscreen.render(640, 480, 0)  # TODO: xtion id is hardcoded
        rgb, depth = self.offscreen.read_pixels(640, 480, depth=True)

        z_near = 0.2 * self.sim.model.stat.extent
        z_far = 50 * self.sim.model.stat.extent
        depth = cv_tools.gl2cv(depth, z_near, z_far)

        # Generate point cloud
        camera_intrinsics = [525, 525, 320, 240]
        point_cloud = cv_tools.depth_to_point_cloud(depth, camera_intrinsics)

        # Get target pose and camera pose
        target_pose = get_body_pose(self.sim, 'target')  # g_wo: object w.r.t. world
        camera_pose = get_camera_pose(self.sim, 'xtion')  # g_wc: camera w.r.t. the world
        camera_to_target = np.matmul(np.linalg.inv(target_pose), camera_pose)  # g_oc = inv(g_wo) * g_wc

        # Transform point cloud w.r.t. to target
        point_cloud = cv_tools.transform_point_cloud(point_cloud, camera_to_target)

        # Keep the points above the table
        points_above_table = []
        for p in point_cloud:
            if p[2] > 0:
                points_above_table.append(p)

        dim = get_geom_size(self.sim.model, 'target')
        points_above_table = np.asarray(points_above_table)
        height_map = cv_tools.generate_height_map(points_above_table)
        features = cv_tools.extract_features(height_map, dim)

        # add the position of the target to the feature
        features.append(target_pose[0, 3])
        features.append(target_pose[1, 3])

        return features

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

        target_size = get_geom_size(self.sim.model, 'target')
        target_pos = self.sim.data.get_joint_qpos('target')
        finger_size = get_geom_size(self.sim.model, 'finger')

        z_off = target_pos[2] + finger_size[0]
        push = Push(direction_theta=action[0], object_height = target_size[0], target=False, z_offset=z_off)

        self.sim.data.set_joint_qpos('finger', [push.initial_pos[0], push.initial_pos[1], z_off + 0.1, 1, 0, 0, 0])
        self.sim_step()

        self.move_joint_to_target('finger', [None, None, push.z])
        end = push.initial_pos + push.distance * push.direction
        self.move_joint_to_target('finger', [end[0], end[1], None])
        self.move_joint_to_target('finger', [None, None, 0.4])

        return time

    def _move_finger_outside_the_table(self):
        # Move finger outside the table again
        table_size = get_geom_size(self.sim.model, 'table')
        table_pos = self.sim.data.get_body_xpos('table')
        f_pos = table_pos + 1.1 * table_size
        self.sim.data.set_joint_qpos('finger', [f_pos[0], f_pos[1], f_pos[2], 1, 0, 0, 0])
        self.sim_step()

    def viewer_setup(self):
        # Set the camera configuration (spherical coordinates)
        self.viewer.cam.distance = 1.2
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
        init_time = self.time
        desired_quat = Quaternion()

        if target_position[0] is not None:
            trajectory_x = Trajectory([self.time, self.time + duration], [self.finger_pos[0], target_position[0]])
        else:
            trajectory_x = Trajectory([self.time, self.time + duration], [self.finger_pos[0], self.finger_pos[0]])
        if target_position[1] is not None:
            trajectory_y = Trajectory([self.time, self.time + duration], [self.finger_pos[1], target_position[1]])
        else:
            trajectory_y = Trajectory([self.time, self.time + duration], [self.finger_pos[1], self.finger_pos[1]])
        if target_position[2] is not None:
            trajectory_z = Trajectory([self.time, self.time + duration], [self.finger_pos[2], target_position[2]])
        else:
            trajectory_z = Trajectory([self.time, self.time + duration], [self.finger_pos[2], self.finger_pos[2]])

        while self.time <= init_time + duration:
            quat_error = self.finger_quat.error(desired_quat)

            # TODO: The indexes of the actuators are hardcoded right now
            # assuming that 0-6 is the actuator of the given joint
            self.sim.data.ctrl[0] = self.pd.get_control(trajectory_x.pos(self.time) - self.finger_pos[0], trajectory_x.vel(self.time) - self.finger_vel[0])
            self.sim.data.ctrl[1] = self.pd.get_control(trajectory_y.pos(self.time) - self.finger_pos[1], trajectory_y.vel(self.time) - self.finger_vel[1])
            self.sim.data.ctrl[2] = self.pd.get_control(trajectory_z.pos(self.time) - self.finger_pos[2], trajectory_z.vel(self.time) - self.finger_vel[2])
            self.sim.data.ctrl[3] = self.pd_rot[0].get_control(quat_error[0], - self.finger_vel[3])
            self.sim.data.ctrl[4] = self.pd_rot[1].get_control(quat_error[1], - self.finger_vel[4])
            self.sim.data.ctrl[5] = self.pd_rot[2].get_control(quat_error[2], - self.finger_vel[5])

            self.sim_step()
            self.render()

            current_pos = self.sim.data.get_joint_qpos(joint_name)

    def sim_step(self):
        """
        A wrapper for sim.step() which updates every time a local state structure.
        """
        self.finger_quat_prev = self.finger_quat

        self.sim.step()

        self.time = self.sim.data.time

        current_pos = self.sim.data.get_joint_qpos("finger")
        self.finger_pos = np.array([current_pos[0], current_pos[1], current_pos[2]])
        self.finger_quat = Quaternion(w=current_pos[3], x=current_pos[4], y=current_pos[5], z=current_pos[6])
        if (np.inner(self.finger_quat.as_vector(), self.finger_quat_prev.as_vector()) < 0):
            self.finger_quat.w = - self.finger_quat.w
            self.finger_quat.x = - self.finger_quat.x
            self.finger_quat.y = - self.finger_quat.y
            self.finger_quat.z = - self.finger_quat.z
        self.finger_quat.normalize()

        self.finger_vel = self.sim.data.get_joint_qvel('finger')

        # Calculate the object's length, width and height w.r.t. the surface by
        # using the orientation of the object. The height is the dimension
        # along the the surface normal. The length is the maximum dimensions
        # between the remaining two.
        target_rot = self.sim.data.get_body_xmat('target')
        if (np.abs(np.inner(target_rot[:, 0], self.surface_normal)) > 0.9):
            self.target_height = self.target_size[0]
            self.target_length = max(self.target_size[1], self.target_size[2])
            self.target_width = min(self.target_size[1], self.target_size[2])
        elif (np.abs(np.inner(target_rot[:, 1], self.surface_normal)) > 0.9):
            self.target_height = self.target_size[1]
            self.target_length = max(self.target_size[0], self.target_size[2])
            self.target_width = min(self.target_size[0], self.target_size[2])
        elif (np.abs(np.inner(target_rot[:, 2], self.surface_normal)) > 0.9):
            self.target_height = self.target_size[2]
            self.target_length = max(self.target_size[0], self.target_size[1])
            self.target_width = min(self.target_size[0], self.target_size[1])




