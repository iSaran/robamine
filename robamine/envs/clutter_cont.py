"""
ClutterCont
=======

Clutter Env for continuous control
"""

import numpy as np
from numpy.linalg import norm

from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjRenderContext
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import os
import gym

from robamine.utils.robotics import PDController, Trajectory
from robamine.utils.mujoco import get_body_mass, get_body_pose, get_camera_pose, get_geom_size, get_body_inertia, get_geom_id, get_body_names, detect_contact, XMLGenerator
from robamine.utils.orientation import Quaternion, rot2angleaxis, rot_z, Affine3
from robamine.utils.math import sigmoid, rescale, filter_signal, LineSegment2D, min_max_scale
import robamine.utils.cv_tools as cv_tools
import math
from math import sqrt
import glfw

from robamine.envs.clutter_utils import TargetObjectConvexHull, InvalidEnvError

import xml.etree.ElementTree as ET
from robamine.utils.orientation import rot2quat

import cv2
from mujoco_py.cymj import MjRenderContext

from time import sleep

import matplotlib.pyplot as plt

def get_observation_dim(primitive, rotations):
    if primitive == 0:
        obs_dim = 659
    elif primitive == 1:
        obs_dim = 405
    elif primitive == 2:
        obs_dim = 400

    if rotations > 0:
        obs_dim *= rotations

    return obs_dim


def exp_reward(x, max_penalty, min, max):
    a = 1
    b = -1.2
    c = -max_penalty
    min_exp = 0.0; max_exp = 5.0
    new_i = rescale(x, min, max, [min_exp, max_exp])
    return max_penalty * a * math.exp(b * new_i) + c

class Primitive:
    def __init__(self, theta, heightmap, mask, pixels_to_m=1):
        self.theta = theta
        self.target_object = TargetObjectConvexHull(cv_tools.Feature(heightmap).mask_in(mask).array()).translate_wrt_centroid().image2world(pixels_to_m)

    def _draw(self):
        fig, ax = self.target_object.draw()

    def plot(self):
        fig, ax = self._draw()
        plt.show()

class Push(Primitive):
    def __init__(self, theta, push_distance, heightmap, mask, push_distance_limits, observation_boundaries, pixels_to_m=1):
        theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-math.pi, math.pi])
        super().__init__(theta=theta_, heightmap=heightmap, mask=mask, pixels_to_m=pixels_to_m)
        self.theta = min_max_scale(theta, range=[-1, 1], target_range=[-math.pi, math.pi])
        self.push_distance = push_distance
        self.target_object = TargetObjectConvexHull(cv_tools.Feature(heightmap).mask_in(mask).array()).translate_wrt_centroid().image2world(pixels_to_m)

        boundaries_points = [np.array([ observation_boundaries[0],  observation_boundaries[1]]),
                             np.array([ observation_boundaries[0], -observation_boundaries[1]]),
                             np.array([-observation_boundaries[0], -observation_boundaries[1]]),
                             np.array([-observation_boundaries[0],  observation_boundaries[1]])]

        self.observation_boundaries = [LineSegment2D(boundaries_points[0], boundaries_points[1]),
                                       LineSegment2D(boundaries_points[1], boundaries_points[2]),
                                       LineSegment2D(boundaries_points[2], boundaries_points[3]),
                                       LineSegment2D(boundaries_points[3], boundaries_points[0])]

        self.push_line_segment = self._get_push_line_segment(push_distance_limits)

    def _get_push_line_segment(self, limits):
        assert limits[0] <= limits[1]
        assert limits[0] >= 0 and limits[1] >= 0
        direction = np.array([math.cos(self.theta + math.pi), math.sin(self.theta + math.pi)])
        line_segment = LineSegment2D(np.zeros(2), 10 * direction)
        min_point = np.zeros(2)
        max_point = line_segment.get_first_intersection_point(self.observation_boundaries)

        min_limit = min(np.linalg.norm(max_point), max(limits[0], np.linalg.norm(min_point)))
        max_limit = max(np.linalg.norm(min_point), min(limits[1], np.linalg.norm(max_point)))
        return LineSegment2D(min_limit * direction, max_limit * direction)

    def get_final_pos(self):
        lambd = min_max_scale(self.push_distance, range=[-1, 1], target_range=[0, 1])
        init = self.push_line_segment.get_point(lambd)
        return np.append(init, self.z)

    def get_duration(self, distance_per_sec = 0.1):
        return np.linalg.norm(self.get_init_pos() - self.get_final_pos()) / distance_per_sec

    def _draw(self):
        fig, ax = self.target_object.draw()

        c = 'red'
        ax.plot( self.push_line_segment.p1[0], self.push_line_segment.p1[1], color=c, marker='o')
        ax.plot( self.push_line_segment.p2[0], self.push_line_segment.p2[1], color=c, marker='.')
        ax.plot([self.push_line_segment.p1[0], self.push_line_segment.p2[0]], [self.push_line_segment.p1[1], self.push_line_segment.p2[1]], color=c, linestyle='-')

        for l in self.observation_boundaries:
            c = 'black'
            ax.plot(l.p1[0], l.p1[1], color=c, marker='o')
            ax.plot(l.p2[0], l.p2[1], color=c, marker='.')
            ax.plot([l.p1[0], l.p2[0]],
                    [l.p1[1], l.p2[1]], color=c, linestyle='-')

        return fig, ax

    def plot(self):
        fig, ax = self._draw()
        plt.show()

class PushTarget(Push):
    def __init__(self, distance, theta, push_distance, heightmap, mask, distance_limits, push_distance_limits, observation_boundaries, finger_size = 0.02, pixels_to_m=1):
        theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-math.pi, math.pi])
        super().__init__(theta=theta_, push_distance=push_distance, heightmap=heightmap, mask=mask,
                         push_distance_limits=push_distance_limits, observation_boundaries=observation_boundaries,
                         pixels_to_m=pixels_to_m)

        self.distance = distance
        self.distance_line_segment = self._get_distance_line_segment(distance_limits, finger_size)
        object_height = self.target_object.get_bounding_box()[2]
        if object_height - finger_size > 0:
            offset = object_height - finger_size
        else:
            offset = 0
        self.z = finger_size + offset + 0.001

    def _get_distance_line_segment(self, limits, finger_size):
        assert limits[0] <= limits[1]
        assert limits[0] >= 0 and limits[1] >= 0
        direction = np.array([math.cos(self.theta), math.sin(self.theta)])
        line_segment = LineSegment2D(np.zeros(2), 10 * direction)
        min_point = line_segment.get_first_intersection_point(self.target_object.convex_hull)
        max_point = line_segment.get_first_intersection_point(self.observation_boundaries)
        min_point += (finger_size + 0.008) * direction

        min_limit = min(np.linalg.norm(max_point), max(limits[0], np.linalg.norm(min_point)))
        max_limit = max(np.linalg.norm(min_point), min(limits[1], np.linalg.norm(max_point)))
        return LineSegment2D(min_limit * direction, max_limit * direction)

    def get_init_pos(self):
        lambd = min_max_scale(self.distance, range=[-1, 1], target_range=[0, 1])
        init = self.distance_line_segment.get_point(lambd)
        return np.append(init, self.z)

    def _draw(self):
        fig, ax = super()._draw()

        c = 'blue'
        ax.plot( self.distance_line_segment.p1[0], self.distance_line_segment.p1[1], color=c, marker='o')
        ax.plot( self.distance_line_segment.p2[0], self.distance_line_segment.p2[1], color=c, marker='.')
        ax.plot([self.distance_line_segment.p1[0], self.distance_line_segment.p2[0]], [self.distance_line_segment.p1[1], self.distance_line_segment.p2[1]], color=c, linestyle='-')

        return fig, ax

class PushObstacle(Push):
    def __init__(self, theta, push_distance, heightmap, mask, observation_boundaries, push_distance_limits, finger_size = 0.02, pixels_to_m=1):
        super().__init__(theta=theta, push_distance=push_distance, heightmap=heightmap, mask=mask,
                         push_distance_limits=push_distance_limits, observation_boundaries=observation_boundaries,
                         pixels_to_m=pixels_to_m)

        self.push_distance = push_distance
        self.theta = min_max_scale(theta, range=[-1, 1], target_range=[-math.pi, math.pi])
        object_height = self.target_object.get_bounding_box()[2]
        self.z = 2 * object_height + finger_size + 0.004
        self.push_line_segment = self._get_push_line_segment(push_distance_limits)

    def get_init_pos(self):
        return np.array([0.0, 0.0, self.z])

class GraspTarget(Primitive):
    def __init__(self, theta, heightmap, mask, finger_size, pixels_to_m):
        theta_ = min_max_scale(theta, range=[-1, 1], target_range=[0, math.pi])
        super().__init__(theta=theta_, heightmap=heightmap, mask=mask, pixels_to_m=pixels_to_m)

        self.grasp_line_segment = self._get_grasp_line_segment(finger_size)

        object_height = self.target_object.get_bounding_box()[2]
        if object_height - finger_size > 0:
            offset = object_height - finger_size
        else:
            offset = 0
        self.z = finger_size + offset + 0.001

    def _get_grasp_line_segment(self, finger_size):
        # Finger 1
        direction = np.array([math.cos(self.theta), math.sin(self.theta)])
        line_segment = LineSegment2D(np.zeros(2), 10 * direction)
        point1 = line_segment.get_first_intersection_point(self.target_object.convex_hull)
        point1 += (finger_size + 0.008) * direction

        # Finger 2
        direction = np.array([math.cos(self.theta + math.pi), math.sin(self.theta + math.pi)])
        line_segment = LineSegment2D(np.zeros(2), 10 * direction)
        point2 = line_segment.get_first_intersection_point(self.target_object.convex_hull)
        point2 += (finger_size + 0.008) * direction
        return LineSegment2D(point1, point2)

    def get_init_pos(self):
        return np.append(self.grasp_line_segment.p1, self.z)

    def get_final_pos(self):
        return np.append(self.grasp_line_segment.p2, self.z)

    def _draw(self):
        fig, ax = self.target_object.draw()

        c = 'red'
        ax.plot( self.grasp_line_segment.p1[0], self.grasp_line_segment.p1[1], color=c, marker='o')
        ax.plot( self.grasp_line_segment.p2[0], self.grasp_line_segment.p2[1], color=c, marker='.')
        ax.plot([self.grasp_line_segment.p1[0], self.grasp_line_segment.p2[0]], [self.grasp_line_segment.p1[1], self.grasp_line_segment.p2[1]], color=c, linestyle='-')

        return fig, ax

class GraspObstacle:
    def __init__(self, theta, distance, phi, spread, height, target_bounding_box, finger_radius):
        self.theta = theta
        self.phi = phi
        self.spread = spread
        self.bb_angle = math.atan2(target_bounding_box[1], target_bounding_box[0])

        theta_ = abs(theta)
        if theta_ > math.pi / 2:
            theta_ = math.pi - theta_

        if theta_ >= self.bb_angle:
            minimum_distance = target_bounding_box[1] / math.sin(theta_)
        else:
            minimum_distance = target_bounding_box[0] / math.cos(theta_)

        self.distance = minimum_distance + finger_radius + distance + 0.005

        object_height = target_bounding_box[2]
        if object_height - finger_radius > 0:
            offset = object_height - finger_radius
        else:
            offset = 0
        self.z = finger_radius + offset + 0.001

    def get_init_pos(self):
        finger_1_init = self.distance * np.array([math.cos(self.theta), math.sin(self.theta)])

        # Finger w.r.t. the position of finger 2
        finger_2_init = self.spread * np.array([math.cos(self.phi), math.sin(self.phi)])


        # Calculate local frame on f1 position in order to avoid target collision
        if abs(self.theta) < self.bb_angle:
            rot = np.array([[ 0, 1],
                            [-1, 0]])
        elif abs(self.theta) < math.pi - self.bb_angle:
            if self.theta > 0:
                rot = np.array([[1, 0],
                                [0, 1]])
            else:
                rot = np.array([[-1,  0],
                                [ 0, -1]])
        else:
            rot = np.array([[0, -1],
                            [1,  0]])

        finger_2_init = np.matmul(rot, finger_2_init) + finger_1_init

        return np.append(finger_1_init, self.z), np.append(finger_2_init, self.z)

class ClutterXMLGenerator(XMLGenerator):
    def __init__(self, path, clutter_params):
        self.params = clutter_params.copy()
        tree = ET.parse(path)
        self.root = tree.getroot()
        for worldbody in self.root:
            if worldbody.tag == 'worldbody':
                break
        self.worldbody = worldbody
        self.rng = np.random.RandomState()  # rng for the scene
        self.n_obstacles = 0


        # Auxiliary variables
        self.surface_size = np.zeros(2)

    def get_object(self, name, type='box', pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0], rgba=[0.0, 0.4, 0.6, 1.0], size=[0.01, 0.01, 0.01]):
        body = self.get_body(name=name, pos=pos, quat=quat)
        joint = self.get_joint(name=name, type='free')
        geom = self.get_geom(name=name, type=type, size=size, rgba=rgba)
        body.append(joint)
        body.append(geom)
        return body

    def get_target(self, type='box', pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0], rgba=[1.0, 0.0, 0.0, 1.0], size=[0.01, 0.01, 0.01]):
        return self.get_object(name='target', type=type, pos=pos, quat=quat, rgba=rgba, size=size)

    def get_obstacle(self, index, type='box', pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0], rgba=[0.0, 0.0, 1.0, 1.0], size=[0.01, 0.01, 0.01]):
        return self.get_object(name='object' + str(index), type=type, pos=pos, quat=quat, rgba=rgba, size=size)

    def get_finger(self, index, type='sphere', rgba=[0.3, 0.3, 0.3, 1.0], size=[0.005, 0.005, 0.005]):
        return self.get_object(name='finger' + str(index), type=type, rgba=rgba, size=size)

    def get_table(self, rgba=[0.2, 0.2, 0.2, 1.0], size=[0.25, 0.25, 0.01]):
        body = self.get_body(name='table', pos=[0.0, 0.0, -size[2]])
        geom = self.get_geom(name='table', type='box', size=size, rgba=rgba)
        body.append(geom)
        return body

    def seed(self, seed):
        self.rng.seed(seed)

    def generate_random_xml(self, surface_length_range=[0.25, 0.25], surface_width_range=[0.25, 0.25]):

        # Setup fingers
        # -------------

        finger_size = self.rng.uniform(self.params['finger_size'][0], self.params['finger_size'][1])
        finger = self.get_finger(1, size=[finger_size, finger_size, finger_size])
        self.worldbody.append(finger)

        finger_size = self.rng.uniform(self.params['finger_size'][0], self.params['finger_size'][1])
        finger = self.get_finger(2, size=[finger_size, finger_size, finger_size])
        self.worldbody.append(finger)

        # Setup surface
        # -------------

        self.surface_size[0] = self.rng.uniform(surface_length_range[0], surface_length_range[1])
        self.surface_size[1] = self.rng.uniform(surface_width_range[0], surface_width_range[1])
        table = self.get_table(size=[self.surface_size[0], self.surface_size[1], 0.01])
        self.worldbody.append(table)

        # Randomize target object
        # -----------------------

        #   Randomize type (box or cylinder)
        temp = self.rng.uniform(0, 1)
        if (temp < self.params['target']['probability_box']):
            type = 'box'
        else:
            type = 'cylinder'
            # # Increase the friction of the cylinders to stabilize them
            # self.sim.model.geom_friction[geom_id][0] = 1.0
            # self.sim.model.geom_friction[geom_id][1] = .01
            # self.sim.model.geom_friction[geom_id][2] = .01
            # self.sim.model.geom_condim[geom_id] = 4
            # self.sim.model.geom_solref[geom_id][0] = .002

        #   Randomize size
        target_length = self.rng.uniform(self.params['target']['min_bounding_box'][0], self.params['target']['max_bounding_box'][0])
        target_width  = self.rng.uniform(self.params['target']['min_bounding_box'][1], min(target_length, self.params['target']['max_bounding_box'][1]))
        target_height = self.rng.uniform(max(self.params['target']['min_bounding_box'][2], finger_size), self.params['target']['max_bounding_box'][2])

        #   Randomize orientation
        # theta = self.rng.uniform(0, 2 * math.pi)
        # target_orientation = Quaternion()
        # target_orientation.rot_z(theta)
        # index = self.sim.model.get_joint_qpos_addr("target")
        # random_qpos[index[0] + 3] = target_orientation.w
        # random_qpos[index[0] + 4] = target_orientation.x
        # random_qpos[index[0] + 5] = target_orientation.y
        # random_qpos[index[0] + 6] = target_orientation.z

        if type == 'box':
            target = self.get_target(type, size = [target_length, target_width, target_height])
        else:
            target = self.get_target(type, size = [target_length, target_height, 0.0])
        self.worldbody.append(target)

        # Randomize obstacles
        # -------------------

        all_equal_height = self.rng.uniform(0, 1)

        if all_equal_height < self.params['all_equal_height_prob']:
            self.n_obstacles = self.params['nr_of_obstacles'][1]
        else:
            self.n_obstacles = self.params['nr_of_obstacles'][0] + self.rng.randint(self.params['nr_of_obstacles'][1] - self.params['nr_of_obstacles'][0] + 1)  # 5 to 25 obstacles

        for i in range(1, self.n_obstacles + 1):
            # Randomize type (box or cylinder)
            temp = self.rng.uniform(0, 1)
            if (temp < self.params['obstacle']['probability_box']):
                type = 'box'
            else:
                type = 'cylinder'
                # Increase the friction of the cylinders to stabilize them
                # self.sim.model.geom_friction[geom_id][0] = 1.0
                # self.sim.model.geom_friction[geom_id][1] = .01
                # self.sim.model.geom_friction[geom_id][2] = .01
                # self.sim.model.geom_condim[geom_id] = 4

            #   Randomize size
            obstacle_length = self.rng.uniform(self.params['obstacle']['min_bounding_box'][0], self.params['obstacle']['max_bounding_box'][0])
            obstacle_width  = self.rng.uniform(self.params['obstacle']['min_bounding_box'][1], min(obstacle_length, self.params['obstacle']['max_bounding_box'][1]))

            if all_equal_height < self.params['all_equal_height_prob']:
                obstacle_height = target_height
            else:
                # obstacle_height = self.rng.uniform(max(self.params['obstacle']['min_bounding_box'][2], finger_height), self.params['obstacle']['max_bounding_box'][2])
                min_h = max(self.params['obstacle']['min_bounding_box'][2], target_height + finger_size)
                if min_h > self.params['obstacle']['max_bounding_box'][2]:
                    obstacle_height = self.params['obstacle']['max_bounding_box'][2]
                else:
                    obstacle_height = self.rng.uniform(min_h, self.params['obstacle']['max_bounding_box'][2])

            if type == 'box':
                x = obstacle_length
                y = obstacle_width
                z = obstacle_height
            else:
                x = obstacle_length
                y = obstacle_height
                z = 0.0

            # Randomize the positions
            r = self.rng.exponential(0.01) + target_length + max(x, y)
            theta = self.rng.uniform(0, 2 * math.pi)
            pos = [r * math.cos(theta), r * math.sin(theta), z]
            obstacle = self.get_obstacle(index=i, type=type, pos=pos, size=[x, y, z])
            self.worldbody.append(obstacle)

        xml = ET.tostring(self.root, encoding="utf-8", method="xml").decode("utf-8")
        return xml

class ClutterContWrapper(gym.Env):
    def __init__(self, params):
        self.params = params
        self.params['seed'] = self.params.get('seed', None)
        self.env = None

        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        self.hardcoded_primitive = self.params.get('hardcoded_primitive', None)
        self.heightmap_rotations = self.params.get('heightmap_rotations', 0)
        obs_dim = get_observation_dim(self.hardcoded_primitive, self.heightmap_rotations)

        self.observation_space = spaces.Box(low=np.full((obs_dim,), 0),
                                            high=np.full((obs_dim,), 0.3),
                                            dtype=np.float32)

    def reset(self, seed=None):
        if self.env is not None:
            del self.env
        self.params['seed'] = seed
        reset_not_valid = True
        while reset_not_valid:
            reset_not_valid = False
            self.env = gym.make('ClutterCont-v0', params=self.params)
            try:
                obs = self.env.reset()
            except InvalidEnvError as e:
                print("WARN: {0}. Invalid environment during reset. A new environment will be spawn.".format(e))
                reset_not_valid = True
        return obs

    def step(self, action):
        try:
            result = self.env.step(action)
        except InvalidEnvError as e:
            print("WARN: {0}. Invalid environment during step. A new environment will be spawn.".format(e))
            self.reset()
            return self.step(action)
        return result

    def seed(self, seed):
        self.env.seed(seed)

    def render(self, mode='human'):
        self.env.render(mode)


class ClutterCont(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    The class for the Gym environment.
    """
    def __init__(self, params):
        self.params = params
        self.log_dir = self.params.get('log_dir', '/tmp')
        path = os.path.join(os.path.dirname(__file__), "assets/xml/robots/clutter.xml")
        self.xml_generator = ClutterXMLGenerator(path, params)
        self.rng = np.random.RandomState()  # rng for the scene
        s = self.params.get('seed', None)
        self.seed(s)

        xml = self.xml_generator.generate_random_xml()

        self.model = load_model_from_xml(xml)
        self.sim = MjSim(self.model)
        self._viewers = {}
        self.viewer = None

        self.offscreen = MjRenderContext(sim=self.sim, device_id=0, offscreen=True, opengl_backend='glfw')

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        self.hardcoded_primitive = self.params.get('hardcoded_primitive', None)
        self.heightmap_rotations = self.params.get('heightmap_rotations', 0)
        obs_dim = get_observation_dim(self.hardcoded_primitive, self.heightmap_rotations)

        self.observation_space = spaces.Box(low=np.full((obs_dim,), 0),
                                            high=np.full((obs_dim,), 0.3),
                                            dtype=np.float32)

        finger_mass = get_body_mass(self.sim.model, 'finger1')
        self.pd = PDController.from_mass(mass = finger_mass)

        moment_of_inertia = get_body_inertia(self.sim.model, 'finger1')
        self.pd_rot = []
        self.pd_rot.append(PDController.from_mass(mass = moment_of_inertia[0], step_response=0.005))
        self.pd_rot.append(PDController.from_mass(mass = moment_of_inertia[1], step_response=0.005))
        self.pd_rot.append(PDController.from_mass(mass = moment_of_inertia[2], step_response=0.005))

        # Parameters, updated once during reset of the model
        self.surface_normal = np.array([0, 0, 1])
        self.surface_size = np.zeros(2)  # half size of the table in x, y
        self.finger_length = 0.0
        self.finger_height = 0.0

        self.target_size = np.zeros(3)

        self.no_of_prev_points_around = 0
        self.prev_point_cloud = []
        # State variables. Updated after each call in self.sim_step()
        self.time = 0.0
        self.timesteps = 0
        self.max_timesteps = self.params.get('max_timesteps', 20)
        self.finger_pos = np.zeros(3)
        self.finger2_pos = np.zeros(3)
        self.finger_quat = Quaternion()
        self.finger2_quat = Quaternion()
        self.finger_quat_prev = Quaternion()
        self.finger2_quat_prev = Quaternion()
        self.finger_vel = np.zeros(6)
        self.finger2_vel = np.zeros(6)
        self.finger_acc = np.zeros(3)
        self.finger2_acc = np.zeros(3)
        self.finger_external_force_norm = 0.0
        self.finger2_external_force_norm = 0.0
        self.finger_external_force = None
        self.finger2_external_force = None
        self.target_pos = np.zeros(3)
        self.target_quat = Quaternion()
        self.push_stopped_ext_forces = False  # Flag if a push stopped due to external forces. This is read by the reward function and penalize the action
        self.last_timestamp = 0.0  # The last time stamp, used for calculating durations of time between timesteps representing experience time
        self.target_grasped_successfully = False
        self.obstacle_grasped_successfully = False
        self.success = False
        self.push_distance = 0.0
        self.grasp_spread = 0.0
        self.grasp_height = 0.0

        self.target_init_pose = Affine3()
        self.predicted_displacement_push_step= np.zeros(3)


        # Initialize this parent class because our environment wraps Mujoco's  C/C++ code.
        utils.EzPickle.__init__(self)
        self.preloaded_init_state = None

        self.pixels_to_m = 0.0012
        self.color_detector = cv_tools.ColorDetector('red')
        fovy = self.sim.model.vis.global_.fovy
        self.size = [640, 480]
        self.camera = cv_tools.PinholeCamera(fovy, self.size)
        self.rgb_to_camera_frame = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

        # Target state from vision
        self.target_bounding_box_vision = np.zeros(3)
        self.target_bounding_box = np.zeros(3)
        self.target_pos_vision = np.zeros(3)
        self.target_quat_vision = Quaternion()

        self.feature_normalization_per = self.params.get('feature_normalization_per', 'session')
        self.max_object_height = 10
        if self.feature_normalization_per == 'session':
            self.max_object_height = 2 * max(max(self.params['target']['max_bounding_box']),
                                             max(self.params['obstacle']['max_bounding_box']))

        self.max_singulation_area = [40, 40]
        self.observation_area = [50, 50]

        self.target_object = None

    def __del__(self):
        if self.viewer is not None:
            glfw.make_context_current(self.viewer.window)
            glfw.destroy_window(self.viewer.window)
        glfw.make_context_current(self.offscreen.opengl_context.window)
        glfw.destroy_window(self.offscreen.opengl_context.window)

    def reset_model(self):
        self.sim_step()
        dims = self.get_object_dimensions('target', self.surface_normal)

        self.timesteps = 0

        if self.preloaded_init_state:
            for i in range(len(self.sim.model.geom_size)):
                self.sim.model.geom_size[i] = self.preloaded_init_state['geom_size'][i]
                self.sim.model.geom_type[i] = self.preloaded_init_state['geom_type'][i]
                self.sim.model.geom_friction[i] = self.preloaded_init_state['geom_friction'][i]
                self.sim.model.geom_condim[i] = self.preloaded_init_state['geom_condim'][i]

            # Set the initial position of the finger outside of the table, in order
            # to not occlude the objects during reading observation from the camera
            index = self.sim.model.get_joint_qpos_addr('finger1')
            qpos = self.preloaded_init_state['qpos'].copy()
            qvel = self.preloaded_init_state['qvel'].copy()
            qpos[index[0]]   = 100
            qpos[index[0]+1] = 100
            qpos[index[0]+2] = 100
            index = self.sim.model.get_joint_qpos_addr('finger2')
            qpos[index[0]]   = 102
            qpos[index[0]+1] = 102
            qpos[index[0]+2] = 102
            self.set_state(qpos, qvel)
            self.push_distance = self.preloaded_init_state['push_distance']
            self.preloaded_init_state = None

            self.sim_step()
        else:
            # random_qpos, number_of_obstacles = self.xmlgenerate_random_scene()
            random_qpos = self.init_qpos.copy()

            number_of_obstacles = self.xml_generator.n_obstacles

            # Randomize pushing distance
            self.push_distance = self.rng.uniform(self.params['push']['distance'][0],
                                                  self.params['push']['distance'][1])
            self.grasp_spread = self.rng.uniform(self.params['grasp']['spread'][0], self.params['grasp']['spread'][1])
            self.grasp_height = self.rng.uniform(self.params['grasp']['height'][0], self.params['grasp']['height'][1])

            # Set the initial position of the finger outside of the table, in order
            # to not occlude the objects during reading observation from the camera
            index = self.sim.model.get_joint_qpos_addr('finger1')
            random_qpos[index[0]]   = 100
            random_qpos[index[0]+1] = 100
            random_qpos[index[0]+2] = 100
            index = self.sim.model.get_joint_qpos_addr('finger2')
            random_qpos[index[0]]   = 102
            random_qpos[index[0]+1] = 102
            random_qpos[index[0]+2] = 102

            self.set_state(random_qpos, self.init_qvel)

            # Move forward the simulation to be sure that the objects have landed
            for _ in range(600):
                self.sim_step()

            self._hug_target(number_of_obstacles)

            self.check_target_occlusion(number_of_obstacles)

            for _ in range(100):
                for i in range(1, number_of_obstacles):
                     body_id = get_body_names(self.sim.model).index("object"+str(i))
                     self.sim.data.xfrc_applied[body_id][0] = 0
                     self.sim.data.xfrc_applied[body_id][1] = 0
                self.sim_step()

        # Update state variables that need to be updated only once
        self.finger_length = get_geom_size(self.sim.model, 'finger1')[0]
        self.finger_height = get_geom_size(self.sim.model, 'finger1')[0]  # same as length, its a sphere
        self.target_bounding_box = get_geom_size(self.sim.model, 'target')
        self.surface_size = np.array([get_geom_size(self.sim.model, 'table')[0], get_geom_size(self.sim.model, 'table')[1]])

        heightmap, mask = self.get_heightmap()
        self.heightmap_prev = heightmap.copy()
        self.mask_prev = mask.copy()

        self.last_timestamp = self.sim.data.time
        self.success = False
        self.target_grasped_successfully = False
        self.obstacle_grasped_successfully = False

        self.preloaded_init_state = None

        if self.feature_normalization_per == 'episode':
            self.max_object_height = np.max(self.heightmap)

        return self.get_obs()

    def _hug_target(self, number_of_obstacles):
        gain = 150

        names = ["target"]
        for i in range(1, number_of_obstacles + 1):
            names.append("object" + str(i))

        for _ in range(300):
            for name in names:
                body_id = get_body_names(self.sim.model).index(name)
                self.sim.data.xfrc_applied[body_id][0] = - gain * self.sim.data.body_xpos[body_id][0]
                self.sim.data.xfrc_applied[body_id][1] = - gain * self.sim.data.body_xpos[body_id][1]
            self.sim_step()

        for name in names:
            body_id = get_body_names(self.sim.model).index(name)
            self.sim.data.xfrc_applied[body_id][0] = 0
            self.sim.data.xfrc_applied[body_id][1] = 0

        all_objects_still = False
        steps = 0
        while not all_objects_still:
            steps += 1
            all_objects_still = True
            for name in names:
                index = self.sim.model.get_joint_qpos_addr(name)
                # if object is above the table
                if self.sim.data.qpos[index[0] + 2] > 0:
                    index = self.sim.model.get_joint_qvel_addr(name)
                    if np.linalg.norm(self.sim.data.qvel[index[0]:index[0]+6]) > 1e-1:
                        all_objects_still = False
                        break
            self.sim_step()
            wait_steps = 1000
            if steps > wait_steps:
                raise InvalidEnvError('Objects still moving after waiting for ' + str(wait_steps) + ' steps.')


    def seed(self, seed=None):
        super().seed(seed)
        self.xml_generator.seed(seed)
        self.rng.seed(seed)

    def get_heightmap(self):
        self._move_finger_outside_the_table()

        # Grab RGB and depth
        empty_grab = True
        counter = 0
        z_near = 0.2 * self.sim.model.stat.extent
        z_far = 50 * self.sim.model.stat.extent
        while empty_grab:
            self.offscreen.render(640, 480, 0)  # TODO: xtion id is hardcoded
            rgb, depth = self.offscreen.read_pixels(640, 480, depth=True)

            # Convert depth (distance from camera) to heightmap (distance from table)
            depth = cv_tools.gl2cv(depth, z_near, z_far)
            max_depth = np.max(depth)
            depth[depth == 0] = max_depth
            heightmap = max_depth - depth

            # In case of empty RGB or Depth create the offscreen context again
            if len(np.argwhere(rgb > 0)) == 0 or len(np.argwhere(abs(heightmap) > 1e-10)) == 0:
                empty_grab = True
                glfw.make_context_current(self.offscreen.opengl_context.window)
                glfw.destroy_window(self.offscreen.opengl_context.window)
                self.offscreen = MjRenderContext(sim=self.sim, device_id=0, offscreen=True, opengl_backend='glfw')
            else:
                empty_grab = False

            bgr = cv_tools.rgb2bgr(rgb)
            cv2.imwrite(os.path.join(self.log_dir, 'bgr.png'), bgr)
            cv2.imwrite(os.path.join(self.log_dir, 'depth.png'), depth)

            if counter > 20:
                raise InvalidEnvError('Failed to grab a non-empty RGB-D image from offscreen after 20 attempts.')

            counter += 1


        # Calculate masks
        color_detector = cv_tools.ColorDetector('red')
        mask = color_detector.detect(bgr)

        obstacle_detector = cv_tools.ColorDetector('blue')
        mask_obstacles = obstacle_detector.detect(bgr)

        cv2.imwrite(os.path.join(self.log_dir, 'initial_mask.png'), mask)
        if len(np.argwhere(mask > 0)) < 200:
            raise InvalidEnvError('Mask is empty during reset. Possible occlusion of the target object.')

        # Calculate the centroid w.r.t. initial image (640x480) in pixels
        target_object = TargetObjectConvexHull(mask)
        centroid_pxl = target_object.centroid.astype(np.int32)
        target_object.plot(blocking=False, path=self.log_dir)

        # Calculate the centroid and the target pos w.r.t. world
        z = depth[centroid_pxl[1], centroid_pxl[0]]
        centroid_image = self.camera.back_project(centroid_pxl, z)
        centroid_camera = np.matmul(self.rgb_to_camera_frame, centroid_image)
        camera_pose = get_camera_pose(self.sim, 'xtion')  # g_wc: camera w.r.t. the world
        self.target_pos_vision = np.matmul(camera_pose, np.array([centroid_camera[0], centroid_camera[1], centroid_camera[2], 1.0]))[:3]
        self.target_pos_vision[2] /= 2.0

        self.heightmap = cv_tools.Feature(heightmap).translate(centroid_pxl[0], centroid_pxl[1]).crop(193, 193).array()
        plt.imsave(os.path.join(self.log_dir, 'heightmap.png'), self.heightmap, cmap='gray', vmin=np.min(self.heightmap), vmax=np.max(self.heightmap))
        self.mask = cv_tools.Feature(mask).translate(centroid_pxl[0], centroid_pxl[1]).crop(193, 193).array()
        plt.imsave(os.path.join(self.log_dir, 'mask.png'), self.mask, cmap='gray', vmin=np.min(self.mask), vmax=np.max(self.mask))
        self.mask_obstacles = cv_tools.Feature(mask_obstacles).translate(centroid_pxl[0], centroid_pxl[1]).crop(193, 193).array()
        self.singulation_area = (np.array([self.target_bounding_box_vision[1] + 0.01,
                                           self.target_bounding_box_vision[0] + 0.01]) / self.pixels_to_m).astype(np.int32)

        self.target_object = TargetObjectConvexHull(cv_tools.Feature(self.heightmap).mask_in(self.mask.astype(np.int8)).array(), log_dir=self.log_dir)
        self.target_bounding_box_vision = self.target_object.get_bounding_box(self.pixels_to_m)

        if self.singulation_area[0] > self.max_singulation_area[0]:
            self.singulation_area[0] = self.max_singulation_area[0]

        if self.singulation_area[1] > self.max_singulation_area[1]:
            self.singulation_area[1] = self.max_singulation_area[1]

        # cv_tools.plot_2d_img(self.heightmap, 'depth')
        # cv_tools.plot_2d_img(self.mask, 'depth')
        # ToDo: How to normalize depth??

        return self.heightmap, self.mask

    def get_obs(self):
        # return self.get_obs_push_obstacle()
        if self.hardcoded_primitive == 0:
            return self.get_obs_push_target()
        elif self.hardcoded_primitive == 1:
            return self.get_obs_push_obstacle()
        elif self.hardcoded_primitive == 2:
            return self.get_obs_grasp_target()

    def get_obs_push_target(self):
        """
        Read depth and extract height map as observation
        :return:
        """

        def get_feature(heightmap, mask, crop_area, max_object_height, angle=0):

            feature = cv_tools.Feature(heightmap)
            feature = feature.rotate(angle)

            thresholded = np.zeros(feature.array().shape)
            threshold = self.target_bounding_box_vision[2] - 1.5 * self.finger_height
            if threshold < 0:
                threshold = 0
            thresholded[feature.array() > threshold] = 1
            feature = cv_tools.Feature(thresholded)

            feature = feature.crop(crop_area[0], crop_area[1])
            feature = feature.pooling().normalize(max_object_height)
            return feature

        # Add the distance of the object from the edge
        distances = [self.surface_size[0] - self.target_pos_vision[0], \
                     self.surface_size[0] + self.target_pos_vision[0], \
                     self.surface_size[1] - self.target_pos_vision[1], \
                     self.surface_size[1] + self.target_pos_vision[1]]
        distances = [x / 0.5 for x in distances]

        self.get_heightmap()

        enforce_convex_hull = self.params['target'].get('enforce_convex_hull', 10)
        target_object = TargetObjectConvexHull(cv_tools.Feature(self.heightmap).mask_in(self.mask).array()).enforce_number_of_points(enforce_convex_hull).translate_wrt_centroid().image2world(self.pixels_to_m)
        convex_hull_points = target_object.get_limits(sorted=True, normalized=True, polar=True).flatten()

        # Use rotated features
        if self.heightmap_rotations > 0:
            features = []
            rot_angle = 360 / self.heightmap_rotations
            for i in range(0, self.heightmap_rotations):
                depth_feature = get_feature(self.heightmap, self.mask, self.observation_area,
                                            self.max_object_height, rot_angle * i).flatten()

                depth_feature = np.concatenate((depth_feature, convex_hull_points))
                for d in distances:
                    depth_feature = np.append(depth_feature, d)

                features.append(depth_feature)

            depth_feature = np.append(features[0], features[1], axis=0)
            for i in range(2, len(features)):
                depth_feature = np.append(depth_feature, features[i], axis=0)

        # Use single feature (one rotation)
        else:
            depth_feature = get_feature(self.heightmap, self.mask, self.observation_area,
                                        self.max_object_height, 0).flatten()

            depth_feature = np.concatenate((depth_feature, convex_hull_points))
            for d in distances:
                depth_feature = np.append(depth_feature, d)

        return depth_feature

    def get_obs_push_obstacle(self):
        """
        Read depth and extract height map as observation
        :return:
        """

        def get_feature(angle=0):
            feature = cv_tools.Feature(self.heightmap).mask_out(self.mask)
            feature = feature.rotate(angle)

            thresholded = np.zeros(feature.array().shape)
            threshold = 2 * self.target_bounding_box_vision[2] + 1.5 * self.finger_height
            thresholded[feature.array() > threshold] = 1

            feature = cv_tools.Feature(thresholded)
            feature = feature.crop(self.max_singulation_area[0], self.max_singulation_area[1])\
                             .pooling()\
                             .normalize(self.max_object_height)
            return feature

        # Add the distance of the object from the edge
        distances = [self.surface_size[0] - self.target_pos_vision[0], \
                     self.surface_size[0] + self.target_pos_vision[0], \
                     self.surface_size[1] - self.target_pos_vision[1], \
                     self.surface_size[1] + self.target_pos_vision[1]]
        distances = [x / 0.5 for x in distances]

        self.get_heightmap()

        # Calculate normalized convex hall area
        target_object = TargetObjectConvexHull(cv_tools.Feature(self.heightmap).mask_in(self.mask).array()).translate_wrt_centroid().image2world(self.pixels_to_m)
        convex_hull_area = target_object.area()
        max_convex_hull_area = self.params['target']['max_bounding_box'][0] * self.params['target']['max_bounding_box'][0]
        convex_hull_area = min_max_scale(convex_hull_area, range=[0, max_convex_hull_area], target_range=[0, 1])
        if convex_hull_area > 1:
            convex_hull_area = 1
        if convex_hull_area < 0:
            convex_hull_area = 0

        # Use rotated features
        if self.heightmap_rotations > 0:
            features = []
            rot_angle = 360 / self.heightmap_rotations
            for i in range(0, self.heightmap_rotations):

                depth_feature = get_feature(rot_angle * i).flatten()
                depth_feature = np.append(depth_feature, convex_hull_area)

                for d in distances:
                    depth_feature = np.append(depth_feature, d)

                features.append(depth_feature)

            depth_feature = np.append(features[0], features[1], axis=0)
            for i in range(2, len(features)):
                depth_feature = np.append(depth_feature, features[i], axis=0)

        # Use single feature (one rotation)
        else:
            depth_feature = get_feature().flatten()
            depth_feature = np.append(depth_feature, convex_hull_area)
            for d in distances:
                depth_feature = np.append(depth_feature, d)

        return depth_feature

    def get_obs_grasp_target(self):
        """
        Read depth and extract height map as observation
        :return:
        """

        def get_feature(heightmap, mask, crop_area, max_object_height, angle=0):

            feature = cv_tools.Feature(heightmap)
            feature = feature.rotate(angle)
            my_mask = cv_tools.Feature(mask).rotate(angle)

            thresholded = np.zeros(feature.array().shape)
            threshold = self.target_bounding_box_vision[2] - 1.5 * self.finger_height
            if threshold < 0:
                threshold = 0
            thresholded[feature.array() > threshold] = 1
            thresholded[my_mask.array() > 0] = -1
            feature = cv_tools.Feature(thresholded)

            feature = feature.crop(crop_area[0], crop_area[1])
            feature = feature.pooling()
            return feature

        self.get_heightmap()

        # enforce_convex_hull = self.params['target'].get('enforce_convex_hull', 10)
        # target_object = TargetObjectConvexHull(cv_tools.Feature(self.heightmap).mask_in(self.mask).array()).enforce_number_of_points(enforce_convex_hull).translate_wrt_centroid().image2world(self.pixels_to_m)
        # convex_hull_points = target_object.get_limits(sorted=True, normalized=True, polar=True).flatten()
        grasping_area = [40, 40]

        # Use rotated features
        if self.heightmap_rotations > 0:
            features = []
            rot_angle = 360 / self.heightmap_rotations
            for i in range(0, self.heightmap_rotations):
                depth_feature = get_feature(self.heightmap, self.mask, grasping_area,
                                            self.max_object_height, rot_angle * i).flatten()

                # depth_feature = np.concatenate((depth_feature, convex_hull_points))
                features.append(depth_feature)

            depth_feature = np.append(features[0], features[1], axis=0)
            for i in range(2, len(features)):
                depth_feature = np.append(depth_feature, features[i], axis=0)

        # Use single feature (one rotation)
        else:
            # Max area with finger is 30, 30
            depth_feature = get_feature(self.heightmap, self.mask, grasping_area,
                                        self.max_object_height, 0).flatten()

            # depth_feature = np.concatenate((depth_feature, convex_hull_points))


        return depth_feature

    def step(self, action):
        action_ = action.copy()

        if action_[0] == 0:
            self.observation_area = [50, 50]

        self.timesteps += 1
        time = self.do_simulation(action_)
        experience_time = time - self.last_timestamp
        self.last_timestamp = time

        obs = self.get_obs()

        reward = self.get_reward(obs, action_)
        # reward = self.get_shaped_reward_obs(obs, pcd, dim)
        # reward = self.get_reward_obs(obs, pcd, dim)

        done = False
        if self.terminal_state(obs):
            done = True

        # Extra data for having pushing distance, theta along with displacements
        # of the target
        extra_data = {'target_init_pose': self.target_init_pose.matrix()}

        self.heightmap_prev = self.heightmap.copy()

        return obs, reward, done, {'experience_time': experience_time, 'success': self.success, 'extra_data': extra_data}

    def do_simulation(self, action):
        primitive = int(action[0])
        if self.hardcoded_primitive is not None:
            primitive = self.hardcoded_primitive
        target_pose = Affine3.from_vec_quat(self.target_pos_vision, self.target_quat_vision)
        self.target_grasped_successfully = False
        self.obstacle_grasped_successfully = False

        if primitive == 0 or primitive == 1:

            # Push target primitive
            if primitive == 0:
                push = PushTarget(theta=action[1],
                                  push_distance=action[2],
                                  distance=action[3],
                                  heightmap=self.heightmap,
                                  mask=self.mask,
                                  observation_boundaries=np.array(self.observation_area) * self.pixels_to_m,
                                  distance_limits=self.params['push']['target_init_distance'],
                                  push_distance_limits=self.params['push']['distance'],
                                  finger_size = self.finger_length,
                                  pixels_to_m=self.pixels_to_m)
            # Push obstacle primitive
            elif primitive == 1:
                push = PushObstacle(theta=action[1],
                                    push_distance=1,  # use maximum distance for now
                                    heightmap=self.heightmap,
                                    mask=self.mask,
                                    observation_boundaries=np.array(self.max_singulation_area) * self.pixels_to_m,
                                    push_distance_limits=self.params['push']['distance'],
                                    finger_size=self.finger_length,
                                    pixels_to_m=self.pixels_to_m)

            # Transform pushing from target frame to world frame

            push_initial_pos_world = push.get_init_pos() + self.target_pos_vision
            push_final_pos_world = push.get_final_pos() + self.target_pos_vision

            init_z = 2 * self.target_bounding_box[2] + 0.05
            self.sim.data.set_joint_qpos('finger1', [push_initial_pos_world[0], push_initial_pos_world[1], init_z, 1, 0, 0, 0])
            self.sim_step()
            duration = push.get_duration()

            if self.move_joint_to_target('finger1', [None, None, push.z], stop_external_forces=True):
                end = push_final_pos_world[:2]
                self.move_joint_to_target('finger1', [end[0], end[1], None], duration)
            else:
                self.push_stopped_ext_forces = True

        elif primitive == 2 or primitive == 3:
            if primitive == 2:
                grasp = GraspTarget(theta=action[1],
                                    heightmap=self.heightmap,
                                    mask=self.mask,
                                    finger_size=self.finger_length,
                                    pixels_to_m=self.pixels_to_m)

            if primitive == 3:
                theta = rescale(action[1], min=-1, max=1, range=[-math.pi, math.pi])
                phi = rescale(action[2], min=-1, max=1, range=[0, math.pi])  # hardcoded, read it from table limits
                distance = rescale(action[3], min=-1, max=1, range=self.params['grasp']['workspace'])  # hardcoded, read it from table limits
                grasp = GraspObstacle(theta=theta, distance=distance, phi=phi, spread=self.grasp_spread, height=self.grasp_height, target_bounding_box = self.target_bounding_box_vision, finger_radius = self.finger_length)

            f1_initial_pos_world = self.target_pos_vision + grasp.get_init_pos()
            f2_initial_pos_world = self.target_pos_vision + grasp.get_final_pos()

            init_z = 2 * self.target_bounding_box[2] + 0.05
            self.sim.data.set_joint_qpos('finger1', [f1_initial_pos_world[0], f1_initial_pos_world[1], init_z, 1, 0, 0, 0])
            self.sim.data.set_joint_qpos('finger2', [f2_initial_pos_world[0], f2_initial_pos_world[1], init_z, 1, 0, 0, 0])
            self.sim_step()

            if self.move_joints_to_target([None, None, grasp.z], [None, None, grasp.z], ext_force_policy='avoid'):
                centroid = (f1_initial_pos_world + f2_initial_pos_world) / 2
                f1f2_dir = (f1_initial_pos_world - f2_initial_pos_world) / np.linalg.norm(f1_initial_pos_world - f2_initial_pos_world)
                f1f2_dir_1 = np.append(centroid[:2] + f1f2_dir[:2] * 1.1 * self.finger_height, grasp.z)
                f1f2_dir_2 = np.append(centroid[:2] - f1f2_dir[:2] * 1.1 * self.finger_height, grasp.z)
                if not self.move_joints_to_target(f1f2_dir_1, f1f2_dir_2, ext_force_policy='stop'):
                    contacts1 = detect_contact(self.sim, 'finger1')
                    contacts2 = detect_contact(self.sim, 'finger2')
                    if len(contacts1) == 1 and len(contacts2) == 1 and contacts1[0] == contacts2[0]:
                        if primitive == 2:
                            self.target_grasped_successfully = True
                        if primitive == 3:
                            self._remove_obstacle_from_table(contacts1[0])
                            self.obstacle_grasped_successfully = True
            else:
                self.push_stopped_ext_forces = True

        else:
            raise ValueError('Clutter: Primitive ' + str(primitive) + ' does not exist.')

        return self.sim.data.time

    def _move_finger_outside_the_table(self):
        # Move finger outside the table again
        table_size = get_geom_size(self.sim.model, 'table')
        self.sim.data.set_joint_qpos('finger1', [100, 100, 100, 1, 0, 0, 0])
        self.sim.data.set_joint_qpos('finger2', [102, 102, 102, 1, 0, 0, 0])
        self.sim_step()

    def _remove_obstacle_from_table(self, obstacle_name):
        self.sim.data.set_joint_qpos(obstacle_name, [0, 0, -0.2, 1, 0, 0, 0])
        self.sim_step()

    def viewer_setup(self):
        # Set the camera configuration (spherical coordinates)
        self.viewer.cam.distance = 0.75
        self.viewer.cam.elevation = -90  # default -90
        self.viewer.cam.azimuth = 90

    def get_reward(self, observation, action):
        if self.hardcoded_primitive == 0:
            reward = self.get_reward_push_target(observation, action)
        elif self.hardcoded_primitive == 1:
            reward = self.get_reward_push_obstacle(observation, action)
        elif self.hardcoded_primitive == 2:
            reward = self.get_reward_grasp_target()
        reward = rescale(reward, -10, 10, range=[-1, 1])
        return reward

    def get_reward_push_target(self, observation, action):
        # Penalize external forces during going downwards
        if self.push_stopped_ext_forces:
            return -10

        if min([observation[-4], observation[-3], observation[-2], observation[-1]]) < 0:
            return -10

        points_around = cv_tools.Feature(self.mask_obstacles).crop(self.singulation_area[0], self.singulation_area[1])\
                                                             .non_zero_pixels()

        extra_penalty = 0
        if int(action[0]) == 0:
            extra_penalty = -rescale(action[3], -1, 1, range=[0, 5])

        extra_penalty += -rescale(action[2], -1, 1, range=[0, 1])

        if points_around < 20:
            return +10 + extra_penalty

        return -1 + extra_penalty

    def get_reward_push_obstacle(self, observation, action):
        # Penalize external forces during going downwards
        if self.push_stopped_ext_forces:
            return -10

        if min([observation[-4], observation[-3], observation[-2], observation[-1]]) < 0:
            return -10

        points_prev = cv_tools.Feature(self.heightmap_prev).mask_out(self.mask_prev)\
                                                           .crop(self.singulation_area[0], self.singulation_area[1])\
                                                           .non_zero_pixels()
        points_cur = cv_tools.Feature(self.heightmap).mask_out(self.mask)\
                                                     .crop(self.singulation_area[0], self.singulation_area[1])\
                                                     .non_zero_pixels()
        points_diff = np.abs(points_prev - points_cur)

        if points_prev == 0:
            points_prev = 1

        # Compute the percentage of the aera that was freed
        free_area = points_diff / points_prev
        reward = rescale(free_area, 0, 1, range=[0, 10])

        extra_penalty = 0
        # penalize pushes that start far from the target object
        # if int(action[0]) == 0:
        #     extra_penalty = -rescale(action[3], -1, 1, range=[0, 5])

        # if int(action[0]) == 0 or int(action[0]) == 1:
        # extra_penalty += -rescale(action[2], -1, 1, range=[0, 1])

        reward = rescale(reward, 0, 10, range=[-10, 10])
        return reward

        # if points_cur < 20:
        #     return 10 + extra_penalty
        # elif points_diff < 20:
        #     return -5
        # else:
        #     return -1 + extra_penalty

    def get_reward_grasp_target(self):
        # Penalize external forces during going downwards
        if self.push_stopped_ext_forces:
            return -10

        if self.target_grasped_successfully:
            return 10

        return 0

    def terminal_state(self, observation):

        if self.timesteps >= self.max_timesteps:
            return True

        # Terminal if collision is detected
        if self.target_grasped_successfully:
            self.success = True
            return True

        # Terminal if collision is detected
        if self.push_stopped_ext_forces:
            self.push_stopped_ext_forces = False
            return True

        # Terminate if the target flips to its side, i.e. if target's z axis is
        # parallel to table, terminate.
        target_z = self.target_quat.rotation_matrix()[:,2]
        world_z = np.array([0, 0, 1])
        if np.dot(target_z, world_z) < 0.9:
            return True

        # If the object has fallen from the table
        if min([observation[-6], observation[-5], observation[-4], observation[-3]]) < 0:
            return True

        if cv_tools.Feature(self.mask_obstacles).crop(self.singulation_area[0], self.singulation_area[1])\
                                                .non_zero_pixels() < 20:
            self.success = True
            return True

        # If the object is free from obstacles around (no points around)
        # if cv_tools.Feature(self.heightmap).mask_out(self.mask)\
        #                                    .crop(self.singulation_area[0], self.singulation_area[1])\
        #                                    .non_zero_pixels() < 20:
        #     self.success = True
        #     return True

        return False

    def move_joint_to_target(self, joint_name, target_position, duration = 1, stop_external_forces=False):
        """
        Generates a trajectory in Cartesian space (x, y, z) from the current
        position of a joint to a target position. If one of the x, y, z is None
        then the joint will not move in this direction. For example:
        target_position = [None, 1, 1] will move along a trajectory in y,z and
        x will remain the same.

        TODO: The indexes of the actuators are hardcoded right now assuming
        that 0-6 is the actuator of the given joint

        Returns whether it the motion completed or stopped due to external
        forces
        """
        init_time = self.time
        desired_quat = Quaternion()
        self.target_init_pose = Affine3.from_vec_quat(self.target_pos_vision, self.target_quat_vision)

        trajectory = [None, None, None]
        for i in range(3):
            if target_position[i] is None:
                target_position[i] = self.finger_pos[i]
            trajectory[i] = Trajectory([self.time, self.time + duration], [self.finger_pos[i], target_position[i]])

        while self.time <= init_time + duration:
            quat_error = self.finger_quat.error(desired_quat)

            # TODO: The indexes of the actuators are hardcoded right now
            # assuming that 0-6 is the actuator of the given joint
            for i in range(3):
                self.sim.data.ctrl[i] = self.pd.get_control(trajectory[i].pos(self.time) - self.finger_pos[i], trajectory[i].vel(self.time) - self.finger_vel[i])
                self.sim.data.ctrl[i + 3] = self.pd_rot[i].get_control(quat_error[i], - self.finger_vel[i + 3])

            self.sim_step()

            if stop_external_forces and (self.finger_external_force_norm > 0.1):
                break

        # If external force is present move away
        if stop_external_forces and (self.finger_external_force_norm > 0.1):
            self.sim_step()
            # Create a new trajectory for moving the finger slightly in the
            # opposite direction to reduce the external forces
            new_trajectory = [None, None, None]
            duration = 0.2
            for i in range(3):
                direction = (target_position - self.finger_pos) / np.linalg.norm(target_position - self.finger_pos)
                new_target = self.finger_pos - 0.01 * direction  # move 1 cm backwards from your initial direction
                new_trajectory[i] = Trajectory([self.time, self.time + duration], [self.finger_pos[i], new_target[i]], [self.finger_vel[i], 0], [self.finger_acc[i], 0])

            # Perform the trajectory
            init_time = self.time
            while self.time <= init_time + duration:
                quat_error = self.finger_quat.error(desired_quat)

                # TODO: The indexes of the actuators are hardcoded right now
                # assuming that 0-6 is the actuator of the given joint
                for i in range(3):
                    self.sim.data.ctrl[i] = self.pd.get_control(new_trajectory[i].pos(self.time) - self.finger_pos[i], new_trajectory[i].vel(self.time) - self.finger_vel[i])
                    self.sim.data.ctrl[i + 3] = self.pd_rot[i].get_control(quat_error[i], - self.finger_vel[i + 3])

                self.sim_step()

            return False

        return True

    def move_joints_to_target(self, target_position, target_position2, duration=1, duration2=1,
                              ext_force_policy = 'avoid', avoid_threshold=0.1, stop_threshold=101.0):
        assert ext_force_policy == 'avoid' or ext_force_policy == 'ignore' or ext_force_policy == 'stop'
        init_time = self.time
        desired_quat = Quaternion()
        self.target_init_pose = Affine3.from_vec_quat(self.target_pos_vision, self.target_quat_vision)

        trajectory = [None, None, None]
        for i in range(3):
            if target_position[i] is None:
                target_position[i] = self.finger_pos[i]
            trajectory[i] = Trajectory([self.time, self.time + duration], [self.finger_pos[i], target_position[i]])

        trajectory2 = [None, None, None]
        for i in range(3):
            if target_position2[i] is None:
                target_position2[i] = self.finger2_pos[i]
            trajectory2[i] = Trajectory([self.time, self.time + duration2], [self.finger2_pos[i], target_position2[i]])

        while self.time <= init_time + duration:
            quat_error = self.finger_quat.error(desired_quat)
            quat_error2 = self.finger2_quat.error(desired_quat)

            # TODO: The indexes of the actuators are hardcoded right now
            # assuming that 0-6 is the actuator of the given joint
            for i in range(3):
                self.sim.data.ctrl[i] = self.pd.get_control(trajectory[i].pos(self.time) - self.finger_pos[i], trajectory[i].vel(self.time) - self.finger_vel[i])
                self.sim.data.ctrl[i + 3] = self.pd_rot[i].get_control(quat_error[i], - self.finger_vel[i + 3])

            for i in range(3):
                self.sim.data.ctrl[i + 6] = self.pd.get_control(trajectory2[i].pos(self.time) - self.finger2_pos[i], trajectory2[i].vel(self.time) - self.finger2_vel[i])
                self.sim.data.ctrl[i + 6 + 3] = self.pd_rot[i].get_control(quat_error2[i], - self.finger2_vel[i + 3])

            self.sim_step()

            if ext_force_policy == 'avoid' and (self.finger_external_force_norm > avoid_threshold or self.finger2_external_force_norm > avoid_threshold):
                break

            if ext_force_policy == 'stop' and (self.finger_external_force_norm > stop_threshold and self.finger2_external_force_norm > stop_threshold):
                break

        if ext_force_policy == 'stop' and (self.finger_external_force_norm > stop_threshold and self.finger2_external_force_norm > stop_threshold):
            return False

        # If external force is present move away
        if ext_force_policy == 'avoid' and (self.finger_external_force_norm > avoid_threshold or self.finger2_external_force_norm > avoid_threshold):
            self.sim_step()
            # Create a new trajectory for moving the finger slightly in the
            # opposite direction to reduce the external forces
            new_trajectory = [None, None, None]
            duration = 0.2
            new_trajectory2 = [None, None, None]
            for i in range(3):
                direction = (target_position - self.finger_pos) / np.linalg.norm(target_position - self.finger_pos)
                new_target = self.finger_pos - 0.01 * direction  # move 1 cm backwards from your initial direction
                new_trajectory[i] = Trajectory([self.time, self.time + duration], [self.finger_pos[i], new_target[i]], [self.finger_vel[i], 0], [self.finger_acc[i], 0])

                direction2 = (target_position2 - self.finger2_pos) / np.linalg.norm(target_position2 - self.finger2_pos)
                new_target2 = self.finger2_pos - 0.01 * direction2  # move 1 cm backwards from your initial direction
                new_trajectory2[i] = Trajectory([self.time, self.time + duration], [self.finger2_pos[i], new_target2[i]], [self.finger2_vel[i], 0], [self.finger2_acc[i], 0])

            # Perform the trajectory
            init_time = self.time
            while self.time <= init_time + duration:
                quat_error = self.finger_quat.error(desired_quat)
                quat_error2 = self.finger2_quat.error(desired_quat)

                # TODO: The indexes of the actuators are hardcoded right now
                # assuming that 0-6 is the actuator of the given joint
                for i in range(3):
                    self.sim.data.ctrl[i] = self.pd.get_control(new_trajectory[i].pos(self.time) - self.finger_pos[i], new_trajectory[i].vel(self.time) - self.finger_vel[i])
                    self.sim.data.ctrl[i + 3] = self.pd_rot[i].get_control(quat_error[i], - self.finger_vel[i + 3])

                    self.sim.data.ctrl[i + 6] = self.pd.get_control(new_trajectory2[i].pos(self.time) - self.finger2_pos[i], new_trajectory2[i].vel(self.time) - self.finger2_vel[i])
                    self.sim.data.ctrl[i + 6 + 3] = self.pd_rot[i].get_control(quat_error2[i], - self.finger2_vel[i + 3])

                self.sim_step()

            return False

        return True

    def sim_step(self):
        """
        A wrapper for sim.step() which updates every time a local state structure.
        """

        if self.params['render']:
            self.render()

        self.finger_quat_prev = self.finger_quat
        self.finger2_quat_prev = self.finger2_quat

        self.sim.step()

        self.time = self.sim.data.time

        current_pos = self.sim.data.get_joint_qpos("finger1")
        self.finger_pos = np.array([current_pos[0], current_pos[1], current_pos[2]])
        self.finger_quat = Quaternion(w=current_pos[3], x=current_pos[4], y=current_pos[5], z=current_pos[6])
        if (np.inner(self.finger_quat.as_vector(), self.finger_quat_prev.as_vector()) < 0):
            self.finger_quat.w = - self.finger_quat.w
            self.finger_quat.x = - self.finger_quat.x
            self.finger_quat.y = - self.finger_quat.y
            self.finger_quat.z = - self.finger_quat.z
        self.finger_quat.normalize()

        current_pos = self.sim.data.get_joint_qpos("finger2")
        self.finger2_pos = np.array([current_pos[0], current_pos[1], current_pos[2]])
        self.finger2_quat = Quaternion(w=current_pos[3], x=current_pos[4], y=current_pos[5], z=current_pos[6])
        if (np.inner(self.finger2_quat.as_vector(), self.finger2_quat_prev.as_vector()) < 0):
            self.finger2_quat.w = - self.finger2_quat.w
            self.finger2_quat.x = - self.finger2_quat.x
            self.finger2_quat.y = - self.finger2_quat.y
            self.finger2_quat.z = - self.finger2_quat.z
        self.finger2_quat.normalize()

        self.finger_vel = self.sim.data.get_joint_qvel('finger1')
        index = self.sim.model.get_joint_qvel_addr('finger1')
        self.finger_acc = np.array([self.sim.data.qacc[index[0]], self.sim.data.qacc[index[0] + 1], self.sim.data.qacc[index[0] + 2]])

        self.finger2_vel = self.sim.data.get_joint_qvel('finger2')
        index = self.sim.model.get_joint_qvel_addr('finger2')
        self.finger2_acc = np.array([self.sim.data.qacc[index[0]], self.sim.data.qacc[index[0] + 1], self.sim.data.qacc[index[0] + 2]])

        finger_geom_id = get_geom_id(self.sim.model, "finger1")
        geom2body = self.sim.model.geom_bodyid[finger_geom_id]
        self.finger_external_force_norm = np.linalg.norm(self.sim.data.cfrc_ext[geom2body])
        # functions that start with 'c' return the rotational part first, so for
        # the force take the second triplet, w.r.t. the world.
        self.finger_external_force = self.sim.data.cfrc_ext[geom2body][3:]

        finger_geom_id = get_geom_id(self.sim.model, "finger2")
        geom2body = self.sim.model.geom_bodyid[finger_geom_id]
        self.finger2_external_force_norm = np.linalg.norm(self.sim.data.cfrc_ext[geom2body])
        # functions that start with 'c' return the rotational part first, so for
        # the force take the second triplet, w.r.t. the world.
        self.finger2_external_force = self.sim.data.cfrc_ext[geom2body][3:]

        # Calculate the object's length, width and height w.r.t. the surface by
        # using the orientation of the object. The height is the dimension
        # along the the surface normal. The length is the maximum dimensions
        # between the remaining two.
        dims = self.get_object_dimensions('target', self.surface_normal)

        temp = self.sim.data.get_joint_qpos('target')
        self.target_pos = np.array([temp[0], temp[1], temp[2]])
        self.target_quat = Quaternion(w=temp[3], x=temp[4], y=temp[5], z=temp[6])


    def check_target_occlusion(self, number_of_obstacles):
        """
        Checks if an obstacle is above the target object and occludes it. Then
        it removes it from the arena.
        """
        for i in range(1, number_of_obstacles):
            body_id = get_body_names(self.sim.model).index("target")
            target_position = np.array([self.sim.data.body_xpos[body_id][0], self.sim.data.body_xpos[body_id][1]])
            body_id = get_body_names(self.sim.model).index("object"+str(i))
            obstacle_position = np.array([self.sim.data.body_xpos[body_id][0], self.sim.data.body_xpos[body_id][1]])

            # Continue if object has fallen off the table
            if self.sim.data.body_xpos[body_id][2] < 0:
                continue

            distance = np.linalg.norm(target_position - obstacle_position)

            target_length = self.get_object_dimensions('target', self.surface_normal)[0]
            obstacle_length = self.get_object_dimensions("object"+str(i), self.surface_normal)[0]
            if distance < 0.6 * (target_length + obstacle_length):
                index = self.sim.model.get_joint_qpos_addr("object"+str(i))
                qpos = self.sim.data.qpos.ravel().copy()
                qvel = self.sim.data.qvel.ravel().copy()
                qpos[index[0] + 2] = - 0.2
                self.set_state(qpos, qvel)

    def get_object_dimensions(self, object_name, surface_normal):
        """
        Returns the object's length, width and height w.r.t. the surface by
        using the orientation of the object. The height is the dimension
        along the the surface normal. The length is the maximum dimensions
        between the remaining two.
        """
        rot = self.sim.data.get_body_xmat(object_name)
        size = get_geom_size(self.sim.model, object_name)
        geom_id = get_geom_id(self.sim.model, object_name)
        length, width, height = 0.0, 0.0, 0.0
        if self.sim.model.geom_type[geom_id] == 6:  # if box
            if (np.abs(np.inner(rot[:, 0], surface_normal)) > 0.9):
                height = size[0]
                length = max(size[1], size[2])
                width = min(size[1], size[2])
            elif (np.abs(np.inner(rot[:, 1], surface_normal)) > 0.9):
                height = size[1]
                length = max(size[0], size[2])
                width = min(size[0], size[2])
            elif (np.abs(np.inner(rot[:, 2], surface_normal)) > 0.9):
                height = size[2]
                length = max(size[0], size[1])
                width = min(size[0], size[1])
        elif self.sim.model.geom_type[geom_id] == 5:  # if cylinder
            if (np.abs(np.inner(rot[:, 2], surface_normal)) > 0.9):
                height = size[1]
                length = size[0]
                width = size[0]
            else:
                height = size[0]
                length = size[1]
                width = size[0]
        else:
            raise RuntimeError("Object is not neither a box or a cylinder")

        return np.array([length, width, height])

    def state_dict(self):
        state = {}
        state['qpos'] = self.sim.data.qpos.ravel().copy()
        state['qvel'] = self.sim.data.qvel.ravel().copy()
        state['geom_size'] = self.sim.model.geom_size.copy()
        state['geom_type'] = self.sim.model.geom_type.copy()
        state['geom_friction'] = self.sim.model.geom_friction.copy()
        state['geom_condim'] = self.sim.model.geom_condim.copy()
        state['push_distance'] = self.push_distance
        return state

    def load_state_dict(self, state):
        if state:
            self.preloaded_init_state = state.copy()
        else:
            state = None
