"""
ClutterCont
=======

Clutter Env for continuous control
"""

import numpy as np
from numpy.linalg import norm

from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjRenderContext
from mujoco_py.builder import MujocoException
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

from robamine.envs.clutter_utils import (TargetObjectConvexHull, get_action_dim, get_observation_dim,
                                         get_distance_of_two_bbox, transform_list_of_points, is_object_above_object,
                                         predict_collision, ObstacleAvoidanceLoss, PushTargetRealWithObstacleAvoidance,
                                         PushTargetReal, PushTargetRealObjectAvoidance, get_table_point_cloud,
                                         PushTargetDepthObjectAvoidance, PushObstacle, SingulationCondition)

from robamine.algo.core import InvalidEnvError

import xml.etree.ElementTree as ET
from robamine.utils.orientation import rot2quat

import cv2
from mujoco_py.cymj import MjRenderContext

from time import sleep

import matplotlib.pyplot as plt
from robamine.utils.cv_tools import Feature

import torch

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
        super().__init__(theta=theta, push_distance=push_distance, heightmap=heightmap, mask=mask,
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

# class PushObstacle(Push):
#     def __init__(self, theta, push_distance, heightmap, mask, observation_boundaries, push_distance_limits, finger_size = 0.02, pixels_to_m=1):
#         super().__init__(theta=theta, push_distance=push_distance, heightmap=heightmap, mask=mask,
#                          push_distance_limits=push_distance_limits, observation_boundaries=observation_boundaries,
#                          pixels_to_m=pixels_to_m)
#
#         self.push_distance = push_distance
#         self.theta = min_max_scale(theta, range=[-1, 1], target_range=[-math.pi, math.pi])
#         object_height = self.target_object.get_bounding_box()[2]
#         self.z = 2 * object_height + finger_size + 0.004
#         self.push_line_segment = self._get_push_line_segment(push_distance_limits)
#
#     def get_init_pos(self):
#         return np.array([0.0, 0.0, self.z])
#
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

    def get_object(self, name, type='box', pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0], rgba=[0.0, 0.4, 0.6, 1.0], size=[0.01, 0.01, 0.01], mass=None, geom_pos=[0.0, 0.0, 0.0]):
        body = self.get_body(name=name, pos=pos, quat=quat)
        joint = self.get_joint(name=name, type='free')
        geom = self.get_geom(name=name, pos=geom_pos, type=type, size=size, rgba=rgba, mass=mass)
        body.append(joint)
        body.append(geom)
        return body

    def get_target(self, type='box', pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0], rgba=[1.0, 0.0, 0.0, 1.0], size=[0.01, 0.01, 0.01]):
        return self.get_object(name='target', type=type, pos=pos, quat=quat, rgba=rgba, size=size, mass=0.05)

    def get_obstacle(self, index, type='box', pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0], rgba=[0.0, 0.0, 1.0, 1.0], size=[0.01, 0.01, 0.01]):
        return self.get_object(name='object' + str(index), type=type, pos=pos, quat=quat, rgba=rgba, size=size, mass=0.05)

    def get_finger(self, index, type='sphere', rgba=[0.3, 0.3, 0.3, 1.0], size=0.005):
        pos_ = [0.0, 0.0, 0.0]
        if type == 'sphere':
            size_ = [size, size, size]
        elif type == 'box':
            height = 0.04
            size_ = [size, size, height]
            pos_ = [0.0, 0.0, height - size]
        return self.get_object(name='finger' + str(index), type=type, rgba=rgba, size=size_, mass=0.1)

    def get_table(self, rgba=[0.2, 0.2, 0.2, 1.0], size=[0.25, 0.25, 0.01], walls=False):
        body = self.get_body(name='table', pos=[0.0, 0.0, -size[2]])
        geom = self.get_geom(name='table', type='box', size=size, rgba=rgba)
        body.append(geom)
        if walls:
            walls_height = 0.03
            walls_width = 0.01
            rgba_ = rgba.copy()
            rgba_[-1] = 0.4
            geom = self.get_geom(name='table_wall_x', type='box', size=[size[0], walls_width, walls_height], rgba=rgba_,
                                 pos=[0, size[1] + walls_width, walls_height])
            body.append(geom)
            geom = self.get_geom(name='table_wall_x2', type='box', size=[size[0], walls_width, walls_height],
                                 rgba=rgba_, pos=[0, -size[1] - walls_width, walls_height])
            body.append(geom)
            geom = self.get_geom(name='table_wall_y', type='box', size=[walls_width, size[1], walls_height], rgba=rgba_,
                                 pos=[size[0] + walls_width, 0, walls_height])
            body.append(geom)
            geom = self.get_geom(name='table_wall_y2', type='box', size=[walls_width, size[1], walls_height],
                                 rgba=rgba_, pos=[-size[0] - walls_width, 0, walls_height])
            body.append(geom)
        return body

    def seed(self, seed):
        self.rng.seed(seed)

    def generate_random_xml(self, surface_length_range=[0.25, 0.25], surface_width_range=[0.25, 0.25]):

        # Setup fingers
        # -------------

        finger_size = self.params['finger']['size']
        finger = self.get_finger(1, type=self.params['finger']['type'], size=self.params['finger']['size'])
        self.worldbody.append(finger)

        finger = self.get_finger(2, type=self.params['finger']['type'], size=self.params['finger']['size'])
        self.worldbody.append(finger)

        # Setup surface
        # -------------

        self.surface_size[0] = self.rng.uniform(surface_length_range[0], surface_length_range[1])
        self.surface_size[1] = self.rng.uniform(surface_width_range[0], surface_width_range[1])
        table = self.get_table(size=[self.surface_size[0], self.surface_size[1], 0.01],
                               walls=self.params.get('walls', False))
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

        # Randomize position
        theta = self.rng.uniform(0, 2 * math.pi)
        table_line_segments = [LineSegment2D(np.array([self.surface_size[0], -self.surface_size[1]]),
                                             np.array([self.surface_size[0], self.surface_size[1]])),
                               LineSegment2D(np.array([self.surface_size[0], self.surface_size[1]]),
                                             np.array([-self.surface_size[0], self.surface_size[1]])),
                               LineSegment2D(np.array([-self.surface_size[0], self.surface_size[1]]),
                                             np.array([-self.surface_size[0], -self.surface_size[1]])),
                               LineSegment2D(np.array([-self.surface_size[0], -self.surface_size[1]]),
                                             np.array([self.surface_size[0], -self.surface_size[1]]))]
        distance_table = np.linalg.norm(
            LineSegment2D(np.zeros(2), np.array([math.cos(theta), math.sin(theta)])).get_first_intersection_point(
                table_line_segments))

        max_distance = distance_table - math.sqrt(math.pow(target_length, 2) + math.pow(target_width, 2))
        distance = min(1, abs(self.rng.normal(0, 0.5))) * max_distance
        target_pos = [distance * math.cos(theta), distance * math.sin(theta), 0.0]

        if not self.params['target'].get('randomize_pos', True):
            target_pos = np.zeros(3)

        #   Randomize orientation
        theta = self.rng.uniform(0, 2 * math.pi)
        quat = Quaternion()
        quat.rot_z(theta)

        if type == 'box':
            target = self.get_target(type, pos=target_pos, quat=[quat.w, quat.x, quat.y, quat.z], size = [target_length, target_width, target_height])
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
            pos = [r * math.cos(theta) + target_pos[0], r * math.sin(theta) + target_pos[1], z]
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
        self.state_dim = get_observation_dim(self.hardcoded_primitive, self.params.get('real_state', False))
        self.action_dim = get_action_dim(self.hardcoded_primitive)

        self.results = {}
        self.results['collisions'] = 0

        self.last_reset_state_dict = None
        self.safe = self.params.get('safe', True)

    def reset(self, seed=None):
        if self.env is not None:
            del self.env
        if seed is not None:
            self.params['seed'] = seed
        if self.safe:
            reset_not_valid = True
            while reset_not_valid:
                reset_not_valid = False
                self.env = ClutterCont(params=self.params)
                try:
                    obs = self.env.reset()
                except InvalidEnvError as e:
                    print("WARN: {0}. Invalid environment during reset. A new environment will be spawn.".format(e))
                    reset_not_valid = True
        else:
            self.env = ClutterCont(params=self.params)
            obs = self.env.reset()

        self.last_reset_state_dict = self.env.state_dict()
        return obs

    def step(self, action):
        if self.safe:
            try:
                result = self.env.step(action)
                self.results['collisions'] += int(result[3]['collision'])
            except InvalidEnvError as e:
                print("WARN: {0}. Invalid environment during step. A new environment will be spawn.".format(e))
                self.reset()
                return self.step(action)
            return result
        else:
            result = self.env.step(action)
            self.results['collisions'] += int(result[3]['collision'])
            return result

    def seed(self, seed):
        self.env.seed(seed)

    def render(self, mode='human'):
        self.env.render(mode)

    def state_dict(self):
        return self.last_reset_state_dict

    def load_state_dict(self, state_dict):
        self.env.load_state_dict(state_dict)
        self.env.reset_model()


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


        self.hardcoded_primitive = self.params.get('hardcoded_primitive', -1)
        self.action_dim = get_action_dim(self.hardcoded_primitive)
        self.state_dim = get_observation_dim(self.hardcoded_primitive, self.params.get('real_state', False))

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

        self.target_distances_from_limits_vision = None
        self.target_distances_from_limits = None

        self.convex_mask = None
        self.singulation_distance = 0.03
        self.obs_dict, self.obs_dict_prev = None, None
        self.bgr = None
        self.obs_avoider = ObstacleAvoidanceLoss(distance_range=self.params['push']['target_init_distance'])

        self.init_distance_from_target = 0.0
        self.raw_depth = None
        self.centroid_pxl = None
        self.singulation_condition = None

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

            hug_probability = self.params.get('hug_probability', 1)
            if self.rng.uniform(0, 1) < hug_probability:
                self._hug_target(number_of_obstacles)

            if not self._target_is_on_table():
                raise InvalidEnvError('Target has fallen off the table during resetting the env.')

            self.check_target_occlusion_2()

            for _ in range(100):
                for i in range(1, number_of_obstacles):
                     body_id = get_body_names(self.sim.model).index("object"+str(i))
                     self.sim.data.xfrc_applied[body_id][0] = 0
                     self.sim.data.xfrc_applied[body_id][1] = 0
                self.sim_step()

        # Update state variables that need to be updated only once
        self.finger_length = get_geom_size(self.sim.model, 'finger1')[0]
        self.finger_height = get_geom_size(self.sim.model, 'finger1')[2]  # same as length, its a sphere
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

        self.singulation_condition = SingulationCondition(self.finger_length, self.pixels_to_m, finger_max_spread=0.1)

        return self.get_obs()

    def _hug_target(self, number_of_obstacles):
        gain = 40

        names = ["target"]
        for i in range(1, number_of_obstacles + 1):
            names.append("object" + str(i))

        centroid = np.zeros(2)
        for name in names:
            body_id = get_body_names(self.sim.model).index(name)
            pos = self.sim.data.body_xpos[body_id]
            centroid += pos[:2]
        centroid /= len(names)

        for _ in range(300):
            for name in names:
                body_id = get_body_names(self.sim.model).index(name)
                self.sim.data.xfrc_applied[body_id][0] = - gain * (self.sim.data.body_xpos[body_id][0] - centroid[0])
                self.sim.data.xfrc_applied[body_id][1] = - gain * (self.sim.data.body_xpos[body_id][1] - centroid[1])
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
        self.throw_fallen_obstacles_away()

        # Grab RGB and depth
        empty_grab = True
        counter = 0
        z_near = 0.2 * self.sim.model.stat.extent
        z_far = 50 * self.sim.model.stat.extent
        while empty_grab:
            self.offscreen.render(640, 480, 0)  # TODO: xtion id is hardcoded
            rgb, depth = self.offscreen.read_pixels(640, 480, depth=True)

            rgb = Feature(rgb).array()

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
            self.bgr = bgr
            cv2.imwrite(os.path.join(self.log_dir, 'bgr.png'), bgr)
            cv2.imwrite(os.path.join(self.log_dir, 'depth.png'), depth)

            if counter > 20:
                raise InvalidEnvError('Failed to grab a non-empty RGB-D image from offscreen after 20 attempts.')

            counter += 1

        self.raw_depth = depth

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
        self.centroid_pxl = centroid_pxl
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

        # Create a convex mask
        mask_points = np.argwhere(self.mask > 0)
        self.convex_mask = np.zeros(self.mask.shape, dtype=np.uint8)
        cv2.fillPoly(self.convex_mask, pts=[self.target_object.get_limits().astype(np.int32)], color=(255, 255, 255))
        plt.imsave(os.path.join(self.log_dir, 'convex_mask.png'), self.convex_mask, cmap='gray', vmin=np.min(self.convex_mask), vmax=np.max(self.convex_mask))
        if len(np.argwhere(self.convex_mask > 0)) < 200:
            raise InvalidEnvError('Convex mask is empty during reset.')

        if self.singulation_area[0] > self.max_singulation_area[0]:
            self.singulation_area[0] = self.max_singulation_area[0]

        if self.singulation_area[1] > self.max_singulation_area[1]:
            self.singulation_area[1] = self.max_singulation_area[1]

        # cv_tools.plot_2d_img(self.heightmap, 'depth')
        # cv_tools.plot_2d_img(self.mask, 'depth')
        # ToDo: How to normalize depth??


        self.target_distances_from_limits_vision = [self.surface_size[0] - self.target_pos_vision[0], \
                                                    self.surface_size[0] + self.target_pos_vision[0], \
                                                    self.surface_size[1] - self.target_pos_vision[1], \
                                                    self.surface_size[1] + self.target_pos_vision[1]]
        self.target_distances_from_limits_vision = [x / 0.5 for x in self.target_distances_from_limits_vision]

        return self.heightmap, self.mask

    def _target_is_on_table(self):
        body_id = get_body_names(self.sim.model).index('target')
        pos = self.sim.data.body_xpos[body_id]
        if pos[2] > 0:
            return True

        print('Target fell off the table.')
        return False

    def throw_fallen_obstacles_away(self):
        names = ["target"]
        n_obstacles = self.xml_generator.n_obstacles
        for i in range(1, n_obstacles + 1):
            names.append("object" + str(i))
        for i in range(n_obstacles + 1):
            body_id = get_body_names(self.sim.model).index(names[i])
            if self.sim.data.body_xpos[body_id][2] < 0:
                self.sim.data.set_joint_qpos(names[i], [100, 100, -100, 1, 0, 0, 0])
        self.sim_step()

    @staticmethod
    def get_obs_shapes():
        """
        Provides the shapes of the observation returned by the env. The shapes should be constants and should not change
        dynamically, because we want them to store them in arrays like h5py.
        """
        max_n_obstacles = 10  # TODO: the maximum possible number of obstacles is hardcoded.
        return {'target_bounding_box': (3,),
                'finger_height': (1,),
                'finger_length': (1,),
                'observation_area': (2,),
                'max_singulation_area': (2,),
                'target_distances_from_limits': (4,),
                'heightmap_mask': (2, 386, 386),
                'surface_size': (2,),
                'target_pos': (3,),
                'object_poses': (max_n_obstacles, 7),
                'object_bounding_box': (max_n_obstacles, 3),
                'object_above_table': (max_n_obstacles,),
                'n_objects': (1,),
                'max_n_objects': (1,),
                'init_distance_from_target': (1,),
                'singulation_distance': (1,),
                'push_distance_range': (2,),
                'surface_edges': (4, 2),
                'raw_depth': (480, 640),
                'centroid_pxl': (2,)}

    def get_obs(self):
        shapes = self.get_obs_shapes()

        # Calculate object poses, bounding boxes and which are above the table
        n_obstacles = self.xml_generator.n_obstacles
        assert n_obstacles <= shapes['object_poses'][0]
        poses, bounding_box, above_table = np.zeros(shapes['object_poses']), np.zeros(
            shapes['object_bounding_box']), np.zeros(shapes['object_above_table'], dtype=np.bool)
        names = ["target"]
        for i in range(1, n_obstacles + 1):
            names.append("object" + str(i))
        for i in range(n_obstacles + 1):
            body_id = get_body_names(self.sim.model).index(names[i])
            poses[i, 0:3] = self.sim.data.body_xpos[body_id]
            poses[i, 3:] = self.sim.data.body_xquat[body_id]
            bounding_box[i, :] = get_geom_size(self.sim.model, names[i])
            if poses[i, 2] > 0:
                above_table[i] = True

        obs_dict = {
            'target_bounding_box': self.target_bounding_box_vision,
            'finger_height': np.array([self.finger_height]),
            'finger_length': np.array([self.finger_length]),
            'observation_area': np.array(self.observation_area),
            'max_singulation_area': np.array(self.max_singulation_area),
            'target_distances_from_limits': np.array(self.target_distances_from_limits_vision),
            'heightmap_mask': np.zeros(shapes['heightmap_mask']),
            'surface_size': self.surface_size.copy(),
            'target_pos': self.target_pos_vision.copy(),
            'object_poses': poses,
            'object_bounding_box': bounding_box,
            'object_above_table': above_table,
            'n_objects': np.array([self.xml_generator.n_obstacles + 1]),
            'max_n_objects': np.array([shapes['object_poses'][0]]),
            'init_distance_from_target': np.array([self.init_distance_from_target]),
            'singulation_distance': np.array([self.singulation_distance]),
            'push_distance_range': np.array(self.params['push']['distance']),
            'surface_edges': np.array([[self.surface_size[0], self.surface_size[1]],
                                     [-self.surface_size[0], self.surface_size[1]],
                                     [self.surface_size[0], -self.surface_size[1]],
                                     [-self.surface_size[0], -self.surface_size[1]]]),
            'raw_depth': np.zeros(shapes['raw_depth']),
            'centroid_pxl': np.zeros(shapes['centroid_pxl'])
        }

        if not self._target_is_on_table():
            for key in list(shapes.keys()):
                assert obs_dict[key].shape == shapes[key]
            if self.obs_dict is not None:
                self.obs_dict_prev = self.obs_dict.copy()
            self.obs_dict = obs_dict
            return obs_dict

        self.get_heightmap()
        obs_dict['heightmap_mask'][0, :] = self.heightmap
        obs_dict['heightmap_mask'][1, :] = self.convex_mask
        obs_dict['raw_depth'] = self.raw_depth
        obs_dict['centroid_pxl'] = self.centroid_pxl

        assert set(obs_dict.keys()) == set(shapes.keys())

        for key in list(shapes.keys()):
            assert obs_dict[key].shape == shapes[key]

        if self.obs_dict is not None:
            self.obs_dict_prev = self.obs_dict.copy()
        self.obs_dict = obs_dict
        return obs_dict

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
        terminal, reason = self.terminal_state(obs)
        if terminal:
            done = True
        self.push_stopped_ext_forces = False

        # Extra data for having pushing distance, theta along with displacements
        # of the target
        extra_data = {'target_init_pose': self.target_init_pose.matrix()}

        self.heightmap_prev = self.heightmap.copy()

        collision = False
        if reason == 'collision':
            collision = True
        return obs, reward, done, {'experience_time': experience_time,
                                   'success': self.success,
                                   'extra_data': extra_data,
                                   'collision': collision}

    def do_simulation(self, action):
        primitive = int(action[0])
        if self.hardcoded_primitive != -1:
            primitive = self.hardcoded_primitive
        target_pose = Affine3.from_vec_quat(self.target_pos_vision, self.target_quat_vision)
        self.target_grasped_successfully = False
        self.obstacle_grasped_successfully = False
        translate_wrt_target = False

        if primitive == 0 or primitive == 1:

            # Push target primitive
            if primitive == 0:
                if self.params.get('real_state', False):
                    # push = PushTargetRealWithObstacleAvoidance(self.obs_dict, action[1], action[2], action[3],
                    #                       self.params['push']['distance'],
                    #                       self.params['push']['target_init_distance'],
                    #                       translate_wrt_target=translate_wrt_target)
                    # push = PushTargetRealCartesian(action[1], action[2], action[3], self.params['push']['distance'],
                    #                                self.params['push']['target_init_distance'][1],
                    #                                object_height=self.target_bounding_box[2],
                    #                                finger_size=self.finger_height)

                    # push = PushTargetReal(theta=action[1], push_distance=action[2], distance=action[3],

                    # push = PushTargetReal(theta=action[1], push_distance=1, distance=action[2],
                    #                       push_distance_range=self.params['push']['distance'],
                    #                       init_distance_range=self.params['push']['target_init_distance'],
                    #                       object_height=self.target_bounding_box[2],
                    #                       finger_size=self.finger_height)
                    # push.translate(self.obs_dict['object_poses'][0, :2])

                    # push = PushTargetRealObjectAvoidance(self.obs_dict, angle=action[1], push_distance=action[2],
                    #                                       push_distance_range=self.params['push']['distance'],
                    #                                       init_distance_range=self.params['push']['target_init_distance'],
                    #                                       finger_size=self.finger_height,
                    #                                       target_height=self.target_bounding_box[2])
                    # self.init_distance_from_target = push.init_distance_from_target
                    # push.translate(self.obs_dict['object_poses'][0, :2])

                    push = PushTargetDepthObjectAvoidance(self.obs_dict, angle=action[1], push_distance=action[2],
                                                          push_distance_range=self.params['push']['distance'],
                                                          init_distance_range=self.params['push']['target_init_distance'],
                                                          finger_length=self.finger_length,
                                                          finger_height=self.finger_height,
                                                          target_height=self.target_bounding_box[2],
                                                          camera=self.camera,
                                                          pixels_to_m=self.pixels_to_m,
                                                          rgb_to_camera_frame=self.rgb_to_camera_frame,
                                                          camera_pose=get_camera_pose(self.sim, 'xtion'))
                    self.init_distance_from_target = push.init_distance_from_target
                    push.translate(self.obs_dict['object_poses'][0, :2])

                else:
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
                                    push_distance=0,  # use maximum distance for now
                                    push_distance_range=self.params['push']['distance'],
                                    object_height=2 * self.target_bounding_box[2],
                                    finger_height=self.finger_height)
                push.translate(self.obs_dict['object_poses'][0, :2])

            # Transform pushing from target frame to world frame

            # In this case the push is wrt target, tarnsform it to world:
            if translate_wrt_target:
                push_initial_pos_world = push.get_init_pos() + self.target_pos_vision
                push_final_pos_world = push.get_final_pos() + self.target_pos_vision
            else:
                push_initial_pos_world = push.get_init_pos()
                push_final_pos_world = push.get_final_pos()

            # Calculate the orientation of the finger based on the direction of the push
            push_direction = (push_final_pos_world - push_initial_pos_world) / np.linalg.norm(push_final_pos_world - push_initial_pos_world)
            x_axis = push_direction
            y_axis = np.matmul(rot_z(np.pi/2), x_axis)
            z_axis = np.array([0, 0, 1])
            rot_mat = np.transpose(np.array([x_axis, y_axis, z_axis]))
            push_quat = Quaternion.from_rotation_matrix(rot_mat)
            push_quat_angle, _ = rot2angleaxis(rot_mat)

            if self.params['push'].get('predict_collision', True):
                if primitive == 0 and predict_collision(obs=self.obs_dict,
                                                        x=push_initial_pos_world[0], y=push_initial_pos_world[1],
                                                        theta=push_quat_angle):
                    self.push_stopped_ext_forces = True
                    print('Collision detected!')
                    return self.sim.data.time

                self.sim.data.set_joint_qpos('finger1', [push_initial_pos_world[0], push_initial_pos_world[1],
                                                         push_initial_pos_world[2] + 0.01, 1, 0, 0, 0])
                self.sim_step()
                # Move very quickly to push.z with trajectory because finger falls a little after the previous step.
                self.move_joint_to_target(joint_name='finger1', target_position=[None, None, push_initial_pos_world[2]],
                                          desired_quat=push_quat, duration=0.1)
                duration = push.get_duration()

                end = push_final_pos_world[:2]
                self.move_joint_to_target(joint_name='finger1', target_position=[end[0], end[1], None],
                                          desired_quat=push_quat, duration=duration)
            else:
                init_z = 2 * self.target_bounding_box[2] + 0.05 + self.finger_height
                self.sim.data.set_joint_qpos('finger1',
                                             [push_initial_pos_world[0], push_initial_pos_world[1],
                                              init_z, push_quat.w, push_quat.x, push_quat.y, push_quat.z])
                self.sim_step()
                duration = push.get_duration()

                if self.move_joint_to_target(joint_name='finger1',
                                             target_position=[None, None, push_initial_pos_world[2]],
                                             desired_quat=push_quat,
                                             stop_external_forces=True):
                    end = push_final_pos_world[:2]
                    self.move_joint_to_target(joint_name='finger1', target_position=[end[0], end[1], None],
                                              desired_quat=push_quat, duration=duration)
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
        if self.params.get('real_state', False):
            if self.hardcoded_primitive == 0:
                reward = self.get_reward_real_state_push_target(observation, action)
            if self.hardcoded_primitive == 1:
                reward = self.get_reward_real_state_push_obstacle(observation, action)
            if self.hardcoded_primitive == -1:
                reward = self.get_reward_real_state_all(observation, action)
        elif self.hardcoded_primitive == 0:
            reward = self.get_reward_push_target(observation, action)
        elif self.hardcoded_primitive == 1:
            reward = self.get_reward_push_obstacle(observation, action)
        elif self.hardcoded_primitive == 2:
            reward = self.get_reward_grasp_target()
        elif self.hardcoded_primitive == -1:
            reward = self.get_reward_all(observation, action)
        reward = rescale(reward, -10, 10, range=[-1, 1])
        return reward

    def get_reward_all(self, observation, action):
        # Penalize external forces during going downwards
        if self.push_stopped_ext_forces:
            return -10

        if self.target_grasped_successfully:
            return 10

        if min(self.target_distances_from_limits) < 0:
            return -10

        extra_penalty = 0
        if int(action[0]) == 0:
            extra_penalty = -rescale(action[3], -1, 1, range=[0, 5])

        extra_penalty += -rescale(action[2], -1, 1, range=[0, 1])

        return -1 + extra_penalty

    def get_reward_push_target(self, observation, action):
        # Penalize external forces during going downwards
        if self.push_stopped_ext_forces:
            return -10

        if min(self.target_distances_from_limits) < 0:
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

        if min(self.target_distances_from_limits) < 0:
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

    def get_reward_real_state_all(self, observation, action):
        if self.push_stopped_ext_forces:
            return -10

        if observation['object_poses'][0][2] < 0:
            return -10

        dist = self.get_real_distance_from_closest_obstacle(observation)

        if dist > self.singulation_distance:
            return 10

        extra_penalty = 0
        if int(action[0]) == 0:
            extra_penalty = -min_max_scale(self.init_distance_from_target, range=[-1, 1], target_range=[0, 2])

        return -1 + extra_penalty

    def get_reward_real_state_push_target(self, observation, action):
        # if self.push_stopped_ext_forces:
        #     max_init_distance = self.params['push']['target_init_distance'][1]
        #     max_obs_bounding_box = np.max(self.params['obstacle']['max_bounding_box'])
        #     preprocessed = preprocess_real_state(self.obs_dict_prev, max_init_distance, max_obs_bounding_box)
        #
        #     # import matplotlib.pyplot as plt
        #     # from mpl_toolkits.mplot3d import Axes3D
        #     # from robamine.utils.viz import plot_boxes, plot_frames
        #     # fig = plt.figure()
        #     # ax = Axes3D(fig)
        #     # state = self.obs_dict_prev
        #     # plot_boxes(state['object_poses'][state['object_above_table']][:, 0:3],
        #     #            state['object_poses'][state['object_above_table']][:, 3:7],
        #     #            state['object_bounding_box'][state['object_above_table']], ax)
        #     # plot_frames(state['object_poses'][state['object_above_table']][:, 0:3],
        #     #             state['object_poses'][state['object_above_table']][:, 3:7], 0.01, ax)
        #     # ax.axis('equal')
        #     # plt.show()
        #     #
        #     # fig = plt.figure()
        #     # ax = Axes3D(fig)
        #     # state = preprocessed
        #     # plot_boxes(state['object_poses'][state['object_above_table']][:, 0:3],
        #     #            state['object_poses'][state['object_above_table']][:, 3:7],
        #     #            state['object_bounding_box'][state['object_above_table']], ax)
        #     # plot_frames(state['object_poses'][state['object_above_table']][:, 0:3],
        #     #             state['object_poses'][state['object_above_table']][:, 3:7], 0.01, ax)
        #     # ax.axis('equal')
        #     # plt.show()
        #
        #     point_cloud = get_table_point_cloud(preprocessed['object_poses'][preprocessed['object_above_table']],
        #                                          preprocessed['object_bounding_box'][preprocessed['object_above_table']],
        #                                          workspace=[max_init_distance, max_init_distance],
        #                                          density=128)
        #     # fig, ax = plt.subplots()
        #     # ax.scatter(point_cloud[:, 0], point_cloud[:, 1])
        #     # plt.show()
        #     # self.obs_avoider.plot(point_cloud)
        #     # plt.show()
        #     obs_avoid_signal = float(self.obs_avoider(torch.FloatTensor(point_cloud.reshape(1, -1, 2)), torch.FloatTensor(action[1:].reshape(1, -1))).detach().cpu().numpy())
        #     return -5 * obs_avoid_signal - 20

        if self.push_stopped_ext_forces:
            return -10

        if observation['object_poses'][0][2] < 0:
            return -10

        dist = self.get_real_distance_from_closest_obstacle(observation)
        if dist > self.singulation_distance:
            return 10

        extra_penalty = 0
        if int(action[0]) == 0:
            extra_penalty = -min_max_scale(self.init_distance_from_target, range=[-1, 1], target_range=[0, 2])

        # Calculate the sum
        def get_distances_in_singulation_proximity(obs):
            poses = obs['object_poses'][obs['object_above_table']]
            bbox = obs['object_bounding_box'][obs['object_above_table']]
            distances_ = np.zeros(len(poses))
            for i in range(len(poses)):
                distances_[i] = get_distance_of_two_bbox(poses[0], bbox[0], poses[i], bbox[i])
            distances_ = distances_[distances_ < self.singulation_distance]
            distances_[distances_ < 0.001] = 0.001
            return 1 / distances_

        distances = get_distances_in_singulation_proximity(self.obs_dict_prev)
        distances_next = get_distances_in_singulation_proximity(self.obs_dict)

        if np.sum(distances_next) < np.sum(distances) - 10:
            return -3 + extra_penalty

        return -8 + extra_penalty

    def get_reward_real_state_push_obstacle(self, observation, action):
        if self.push_stopped_ext_forces:
            return -10

        dist = self.get_real_distance_from_closest_obstacle(observation)
        if dist > self.singulation_distance:
            return 10

        # Calculate the sum
        def get_distances_in_singulation_proximity(obs):
            poses = obs['object_poses'][obs['object_above_table']]
            bbox = obs['object_bounding_box'][obs['object_above_table']]
            distances_ = np.zeros(len(poses))
            for i in range(len(poses)):
                distances_[i] = get_distance_of_two_bbox(poses[0], bbox[0], poses[i], bbox[i])
            distances_ = distances_[distances_ < self.singulation_distance]
            distances_[distances_ < 0.001] = 0.001
            return 1 / distances_

        distances = get_distances_in_singulation_proximity(self.obs_dict_prev)
        distances_next = get_distances_in_singulation_proximity(self.obs_dict)

        if np.sum(distances_next) < np.sum(distances) - 10:
            return -5

        return -10

    def terminal_state_real_state_push_target(self, obs):
        if self.timesteps >= self.max_timesteps:
            return True, 'timesteps'

        # Terminal if collision is detected
        if self.push_stopped_ext_forces:
            self.push_stopped_ext_forces = False
            return True, 'collision'

        # Terminate if the target flips to its side, i.e. if target's z axis is
        # parallel to table, terminate.
        target_z = self.target_quat.rotation_matrix()[:,2]
        world_z = np.array([0, 0, 1])
        if np.dot(target_z, world_z) < 0.9:
            return True, 'flipped'

        # If the object has fallen from the table
        if obs['object_poses'][0][2] < 0:
            return True, 'fallen'

        # if self.singulation_condition(self.heightmap, self.mask):
        if self.get_real_distance_from_closest_obstacle(obs) > self.singulation_distance:
            self.success = True
            return True, 'singulation'

        return False, ''

    def get_real_distance_from_closest_obstacle(self, obs_dict):
        '''Returns the minimum distance of target from the obstacles'''
        n_objects = int(obs_dict['n_objects'])
        target_pose = obs_dict['object_poses'][0]
        target_bbox = obs_dict['object_bounding_box'][0]

        distances = 100 * np.ones((int(n_objects),))
        for i in range(1, n_objects):
            obstacle_pose = obs_dict['object_poses'][i]
            obstacle_bbox = obs_dict['object_bounding_box'][i]

            distances[i] = get_distance_of_two_bbox(target_pose, target_bbox, obstacle_pose, obstacle_bbox)

        return np.min(distances)

    def terminal_state(self, observation):
        if self.params.get('real_state', False):
            return self.terminal_state_real_state_push_target(observation)
        else:
            return self.terminal_state_visual(observation)

    def terminal_state_visual(self, observation):
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
        if min(self.target_distances_from_limits) < 0:
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

    def move_joint_to_target(self, joint_name, target_position, desired_quat, duration = 1, stop_external_forces=False):
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

            # Overwrite orientation of the finger to avoid the initial instability in orientation control
            current_qpos = self.sim.data.qpos
            index = self.sim.model.get_joint_qpos_addr('finger1')
            current_qpos[index[0] + 3] = desired_quat.w
            current_qpos[index[0] + 4] = desired_quat.x
            current_qpos[index[0] + 5] = desired_quat.y
            current_qpos[index[0] + 6] = desired_quat.z
            self.sim_step()

            if stop_external_forces and (self.finger_external_force_norm > 0.01):
                break

        # If external force is present move away
        if stop_external_forces and (self.finger_external_force_norm > 0.01):
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

                current_qpos = self.sim.data.qpos
                index = self.sim.model.get_joint_qpos_addr('finger1')
                current_qpos[index[0] + 3] = desired_quat.w
                current_qpos[index[0] + 4] = desired_quat.x
                current_qpos[index[0] + 5] = desired_quat.y
                current_qpos[index[0] + 6] = desired_quat.z
                self.sim_step()

            return False

        return True

    def move_joints_to_target(self, target_position, target_position2, duration=1, duration2=1,
                              ext_force_policy = 'avoid', avoid_threshold=0.1, stop_threshold=5):
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

        try:
            self.sim.step()
        except MujocoException as e:
            raise InvalidEnvError(e)

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

        self.target_distances_from_limits = [self.surface_size[0] - self.target_pos[0], \
                                             self.surface_size[0] + self.target_pos[0], \
                                             self.surface_size[1] - self.target_pos[1], \
                                             self.surface_size[1] + self.target_pos[1]]
        self.target_distances_from_limits = [x / 0.5 for x in self.target_distances_from_limits]

        # Check if any object jumped
        names = ['target']
        for i in range(1, self.xml_generator.n_obstacles + 1):
            names.append("object" + str(i))
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if self.sim.model.geom_id2name(contact.geom1) == 'table' and self.sim.model.geom_id2name(
                    contact.geom2) in names:
                threshold = 0.01
                if abs(contact.dist) > threshold:
                    raise InvalidEnvError(
                        'Contact dist of ' + self.sim.model.geom_id2name(contact.geom2) + ' with table: ' + str(
                            contact.dist) + ' > ' + str(threshold) + '. Probably jumped. Timestep: ' + str(
                            self.sim.data.time))


    def check_target_occlusion(self, number_of_obstacles):
        """
        Checks if an obstacle is above the target object and occludes it. Then
        it removes it from the arena.

        Obsolete by check_target_occlusion_2
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

    def check_target_occlusion_2(self, eps=0.003):
        """
        Checks if an obstacle is above the target object and occludes it. Then
        it removes it from the arena.
        """
        n_obstacles = self.xml_generator.n_obstacles
        names = []
        for i in range(n_obstacles):
            names.append("object" + str(i + 1))

        body_id = get_body_names(self.sim.model).index('target')
        target_pose = np.zeros(7)
        target_pose[0:3] = self.sim.data.body_xpos[body_id]
        target_pose[3:] = self.sim.data.body_xquat[body_id]
        target_bbox = get_geom_size(self.sim.model, 'target') - eps

        for i in range(n_obstacles):
            body_id = get_body_names(self.sim.model).index(names[i])
            obstacle_pose = np.zeros(7)
            obstacle_pose[0:3] = self.sim.data.body_xpos[body_id]
            obstacle_pose[3:] = self.sim.data.body_xquat[body_id]
            obstacle_bbox = get_geom_size(self.sim.model, names[i])

            if obstacle_pose[2] < 0:
                continue

            if is_object_above_object(target_pose, target_bbox, obstacle_pose, obstacle_bbox):
                index = self.sim.model.get_joint_qpos_addr(names[i])
                qpos = self.sim.data.qpos.ravel().copy()
                qvel = self.sim.data.qvel.ravel().copy()
                qpos[index[0] + 2] = - 0.2
                self.set_state(qpos, qvel)
