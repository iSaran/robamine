"""
ClutterCont
=======

Clutter Env for continuous control
"""

import numpy as np
from math import cos, sin, pi, acos, atan2, ceil

from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from robamine.utils.math import (LineSegment2D, triangle_area, min_max_scale, cartesian2spherical,
                                 get_centroid_convex_hull)
from robamine.utils.orientation import (rot_x, rot_z, rot2angleaxis, Quaternion, rot_y, transform_poses,
                                        transform_points)
from robamine.utils import cv_tools
from robamine.utils.info import get_now_timestamp
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from robamine.algo.core import InvalidEnvError

from scipy.spatial.distance import cdist

from robamine.algo.util import Transition
import torch
import torch.nn as nn
import time

INFO = True
DEBUG = False
def debug(*args):
    if DEBUG:
        print("DEBUG:clutter_utils:" + " ".join(map(str, args)))
def info(*args):
    if INFO:
        print("INFO:clutter_utils:" + " ".join(map(str, args)))

class TargetObjectConvexHull:
    def __init__(self, masked_in_depth, log_dir='/tmp'):
        self.masked_in_depth = masked_in_depth
        self.intersection = None
        self.mask_shape = masked_in_depth.shape

        self.mask_points = np.argwhere(np.transpose(masked_in_depth) > 0)

        self.log_dir = log_dir


        # Calculate convex hull
        try:
            self.convex_hull, hull_points = self._get_convex_hull()
        except Exception as err:
            import sys
            np.set_printoptions(threshold=sys.maxsize)
            print('Mask points given to convex hull:')
            print(self.mask_points)
            print('max of mask', np.max(masked_in_depth))
            print('min of mask', np.min(masked_in_depth))
            plt.imsave(os.path.join(self.log_dir, 'convex_hull_error_mask.png'), masked_in_depth, cmap='gray', vmin=np.min(masked_in_depth), vmax=np.max(masked_in_depth))
            print(err)
            raise InvalidEnvError('Failed to create convex hull. Convex hull saved to:' + os.path.join(self.log_dir, 'convex_hull_error_mask.png'))

        # Calculate centroid
        self.centroid = self._get_centroid_convex_hull(hull_points)

        # Calculate height
        self.height = np.mean(masked_in_depth[masked_in_depth > 0])

        # Flags
        self.translated = False
        self.moved2world = False

    def get_limits(self, sorted=False, normalized=False, polar=False):
        # Calculate the limit points
        limits = np.zeros((len(self.convex_hull), 2))
        for i in range(len(self.convex_hull)):
            limits[i, :] = self.convex_hull[i].p1.copy()

        if polar:
            polar_limits = np.zeros(limits.shape)
            for i in range(len(limits)):
                polar_limits[i, 0] = np.linalg.norm(limits[i, :])
                polar_limits[i, 1] = atan2(limits[i, 0], limits[i, 1])
            limits = polar_limits

        if sorted:
            limits = limits[np.argsort(limits[:, 0])]

        if normalized:
            if polar:
                limits[:, 0] = min_max_scale(limits[:, 0], range=[0, np.linalg.norm(self.mask_shape)], target_range=[0, 1])
                limits[:, 1] = min_max_scale(limits[:, 1], range=[-pi, pi], target_range=[0, 1])
            else:
                limits[:, 0] = min_max_scale(limits[:, 0], range=[-self.mask_shape[0], self.mask_shape[0]], target_range=[0, 1])
                limits[:, 1] = min_max_scale(limits[:, 1], range=[-self.mask_shape[1], self.mask_shape[1]], target_range=[0, 1])

            limits[limits > 1] = 1
            limits[limits < 0] = 0

        return limits

    def image2world(self, pixels_to_m=1):
        if self.moved2world:
            return self

        rotation = np.array([[1, 0, 0],
                             [0, -1, 0],
                             [0, 0, -1]])
        temp = np.zeros((self.mask_points.shape[0], 3))
        temp[:, :2] = self.mask_points
        self.mask_points = np.transpose(np.matmul(rotation, np.transpose(temp)))[:, :2]
        self.mask_points = pixels_to_m * self.mask_points

        # rotate line segments
        for line_segment in self.convex_hull:
            temp = np.zeros(3)
            temp[0:2] = line_segment.p1
            line_segment.p1 = np.matmul(rotation, temp)[:2]
            line_segment.p1 *= pixels_to_m

            temp = np.zeros(3)
            temp[0:2] = line_segment.p2
            line_segment.p2 = np.matmul(rotation, temp)[:2]
            line_segment.p2 *= pixels_to_m

        temp = np.zeros(3)
        temp[0:2] = self.centroid
        self.centroid = np.matmul(rotation, temp)[:2]
        self.centroid *= pixels_to_m

        self.moved2world = True

        return self

    def translate_wrt_centroid(self):
        if self.translated:
            return self

        self.mask_points = self.mask_points + \
                           np.repeat(-self.centroid.reshape((1, 2)), self.mask_points.shape[0], axis=0)

        for line_segment in self.convex_hull:
            line_segment.p1 -= self.centroid
            line_segment.p2 -= self.centroid

        self.centroid = np.zeros(2)

        self.translated = True

        return self

    def _get_convex_hull(self):
        hull = ConvexHull(self.mask_points)
        hull_points = np.zeros((len(hull.vertices), 2))
        convex_hull = []
        hull_points[0, 0] = self.mask_points[hull.vertices[0], 0]
        hull_points[0, 1] = self.mask_points[hull.vertices[0], 1]
        i = 1
        for i in range(1, len(hull.vertices)):
            hull_points[i, 0] = self.mask_points[hull.vertices[i], 0]
            hull_points[i, 1] = self.mask_points[hull.vertices[i], 1]
            convex_hull.append(LineSegment2D(hull_points[i - 1, :], hull_points[i, :]))
        convex_hull.append(LineSegment2D(hull_points[i, :], hull_points[0, :]))

        return convex_hull, hull_points

    def _get_centroid_convex_hull(self, hull_points):
        tri = Delaunay(hull_points)
        triangles = np.zeros((tri.simplices.shape[0], 3, 2))
        for i in range(len(tri.simplices)):
            for j in range(3):
                triangles[i, j, 0] = hull_points[tri.simplices[i, j], 0]
                triangles[i, j, 1] = hull_points[tri.simplices[i, j], 1]

        centroids = np.mean(triangles, axis=1)

        triangle_areas = np.zeros(len(triangles))
        for i in range(len(triangles)):
            triangle_areas[i] = triangle_area(triangles[i, :, :])

        weights = triangle_areas / np.sum(triangle_areas)

        centroid = np.average(centroids, axis=0, weights=weights)

        return centroid

    def get_limit_intersection(self, theta):
        """theta in rad"""

        self.intersection = None
        max_distance = np.zeros(2)
        max_distance[0] = 1.5 * np.max(self.mask_points[:, 0])
        max_distance[1] = 1.5 * np.max(self.mask_points[:, 1])
        max_distance_norm = np.linalg.norm(max_distance)
        final = max_distance_norm * np.array([cos(theta), sin(theta)])
        push = LineSegment2D(self.centroid, self.centroid + final)

        for h in self.convex_hull:
            self.intersection = push.get_intersection_point(h)
            if self.intersection is not None:
                break

        # If no intersection found log some debugging messages/plots and then raise invalid env error
        if self.intersection is None:
            path = os.path.join(self.log_dir, 'error_no_intersection_' + get_now_timestamp())
            os.makedirs(path)
            np.savetxt(os.path.join(path, 'max_distance.txt'), max_distance, delimiter=',')
            np.savetxt(os.path.join(path, 'push.txt'), push.array(), delimiter=',')
            fig, ax = self.draw()
            ax.plot(push.p1[0], push.p1[1], color='black', marker='o')
            ax.plot(push.p2[0], push.p2[1], color='black', marker='.')
            ax.plot([push.p1[0], push.p2[0]], [push.p1[1], push.p2[1]], color='black', linestyle='-')
            fig.savefig(os.path.join(path, 'target_object.png'))
            plt.close(fig)
            plt.imsave(os.path.join(path, 'mask.png'), self.masked_in_depth, cmap='gray', vmin=np.min(self.masked_in_depth), vmax=np.max(self.masked_in_depth))
            raise InvalidEnvError('No intersection was found. Error report in ' + path)

        return self.intersection, np.linalg.norm(self.centroid - self.intersection)

    def get_pose(self):
        homog = np.eye(4)
        homog[0:2][3] = self.centroid
        return homog

    def draw(self):
        fig, ax = plt.subplots()
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(self.convex_hull))))
        ax.plot(self.mask_points[:, 0], self.mask_points[:, 1], '.', c='lightgrey')

        for line_segment in self.convex_hull:
            c = next(color)

            ax.plot(line_segment.p1[0], line_segment.p1[1], color=c, marker='o')
            ax.plot(line_segment.p2[0], line_segment.p2[1], color=c, marker='.')
            ax.plot([line_segment.p1[0], line_segment.p2[0]], [line_segment.p1[1], line_segment.p2[1]],
                     color=c, linestyle='-')

        ax.plot(self.centroid[0], self.centroid[1], color='black', marker='o')
        ax.axis('equal')

        if self.intersection is not None:
            ax.plot(self.intersection[0], self.intersection[1], color='black', marker='o')

        return fig, ax

    def plot(self, blocking=True, path=None):
        fig, ax = self.draw()
        if blocking:
            plt.show()
        else:
            plt.draw()

        if path is not None:
            fig.savefig(os.path.join(path, 'target_object.png'))
            plt.close(fig)

    def get_bounding_box(self, pixels_to_m=1):
        limits = self.get_limits(sorted=False, normalized=False)
        if not self.translated:
            limits += np.repeat(-self.centroid.reshape((1, 2)), limits.shape[0], axis=0)

        bb = np.zeros(3)
        bb[0] = pixels_to_m * np.max(np.abs(limits[:, 0]))
        bb[1] = pixels_to_m * np.max(np.abs(limits[:, 1]))
        bb[2] = self.height / 2.0
        return bb

    def enforce_number_of_points(self, n_points):
        diff = len(self.convex_hull) - n_points

        # Remove points
        if diff > 0:
            for i in range(diff):
                lenghts = []
                for lin in self.convex_hull:
                    lenghts.append(np.linalg.norm(lin.norm()))
                lenghts[-1] += lenghts[0]
                for i in range(len(lenghts) - 1):
                    lenghts[i] += lenghts[i + 1]

                first_index = np.argsort(lenghts)[0]
                second_index = first_index + 1
                if first_index == len(self.convex_hull) - 1:
                    second_index = 0

                self.convex_hull[second_index] = LineSegment2D(self.convex_hull[first_index].p1, self.convex_hull[second_index].p2)
                self.convex_hull.pop(first_index)

        # Add more points
        elif diff < 0:
            for i in range(abs(diff)):
                lenghts = []
                for lin in self.convex_hull:
                    lenghts.append(np.linalg.norm(lin.norm()))

                index = np.argsort(lenghts)[::-1][0]
                centroid = (self.convex_hull[index].p1 + self.convex_hull[index].p2) / 2.0
                new = LineSegment2D(centroid, self.convex_hull[index].p2)
                self.convex_hull[index] = LineSegment2D(self.convex_hull[index].p1, centroid)
                self.convex_hull.insert(index + 1, new)
        return self

    def area(self):
        '''See https://en.wikipedia.org/wiki/Shoelace_formula'''
        limits = self.get_limits()
        x = limits[:, 0]
        y = limits[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def rotated_to_regular_transitions(transitions, heightmap_rotations=0):

    if heightmap_rotations <= 0:
        return transitions

    resulted_transitions = []
    for transition in transitions:
        state_split = np.split(transition.state, heightmap_rotations)
        next_state_split = np.split(transition.next_state, heightmap_rotations)

        for j in range(heightmap_rotations):

            # actions are btn -1, 1. Change the 1st action which is the angle w.r.t. the target:
            act = transition.action.copy()
            act[1] += j * (2 / heightmap_rotations)
            if act[1] > 1:
                act[1] = -1 + abs(1 - act[1])

            tran = Transition(state=state_split[j].copy(),
                              action=act.copy(),
                              reward=transition.reward,
                              next_state=next_state_split[j].copy(),
                              terminal=transition.terminal)

            resulted_transitions.append(tran)
    return resulted_transitions


def get_action_dim(primitive):
    action_dim_all = [2, 1]

    if primitive >= 0:
        action_dim = [action_dim_all[primitive]]
    else:
        action_dim = action_dim_all

    return action_dim


def get_observation_dim(primitive, real_state=False):
    if real_state:
        obs_dim_all = [RealState.dim(), RealState.dim()]
    else:
        obs_dim_all = [PushTargetFeature.dim(), PushObstacleFeature.dim()]

    if primitive >= 0:
        obs_dim = [obs_dim_all[primitive]]
    else:
        obs_dim = obs_dim_all

    return obs_dim


# State: Features for different primitives
# ----------------------------------------

class Feature:
    def __init__(self, name=None):
        self.name = name

    def rotate(self, angle):
        raise NotImplementedError()

    def array(self):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()


class RealState(Feature):
    def __init__(self, obs_dict, angle=0, sort=True, normalize=True, spherical=False, range_norm=[-1, 1],
                 translate_wrt_target=False, name=None):
        '''Angle in rad.'''
        self.poses = obs_dict['object_poses'][obs_dict['object_above_table']]
        self.bounding_box = obs_dict['object_bounding_box'][obs_dict['object_above_table']]
        self.n_objects = self.poses.shape[0]
        self.max_n_objects = obs_dict['max_n_objects']
        self.range_norm = range_norm
        self.translate_wrt_target = translate_wrt_target

        # Calculate principal corners for feature
        self.principal_corners = np.zeros((self.n_objects, 4, 3))
        self.calculate_principal_corners()
        self.rotate(angle)
        self.principal_corners_plot = self.principal_corners.copy()
        self.surface_size = obs_dict['surface_size']
        self.surface_edges = obs_dict['surface_edges']
        self.coordinates = 'cartesian'
        # Append for table limits
        rot_2d = np.array([[cos(angle), -sin(angle)],
                           [sin(angle), cos(angle)]])
        self.surface_edges = np.transpose(np.matmul(rot_2d, np.transpose(self.surface_edges)))
        if 'surface_distances' in obs_dict:
            self.surface_distances = obs_dict['surface_distances']
        else:
            self.surface_distances = np.zeros(4)

        self.init_distance_from_target = obs_dict['init_distance_from_target'][0]

        if spherical:
            self.coordinates = 'spherical'
            init_shape = self.principal_corners.shape
            self.principal_corners = cartesian2spherical(self.principal_corners.reshape(-1, 3)).reshape(init_shape)
            self.surface_edges[:, 0] = np.sqrt(self.surface_edges[:, 0] ** 2 + self.surface_edges[:, 1] ** 2)
            self.surface_edges[:, 1] = np.arctan2(self.surface_edges[:, 1], self.surface_edges[:, 0])

        if sort:
            self.sort()

        if normalize:
            self.normalize()

    def calculate_principal_corners(self):
        self.principal_corners[0] = self._get_principal_corner_target()
        for i in range(1, self.n_objects):
            self.principal_corners[i, :, :] = self.get_principal_corner_obstacle(i)

    def get_principal_corner_obstacle(self, i):
        pos_target = self.poses[0, 0:3].copy()
        pos_target[2] = 0.0
        pos = self.poses[i, 0:3]
        quat = Quaternion(self.poses[i, 3], self.poses[i, 4], self.poses[i, 5], self.poses[i, 6])
        bbox=self.bounding_box[i]

        # Bounding box corners w.r.t. the object
        bbox_corners_object = np.array([[bbox[0], bbox[1], bbox[2]],
                                        [bbox[0], -bbox[1], bbox[2]],
                                        [bbox[0], bbox[1], -bbox[2]],
                                        [bbox[0], -bbox[1], -bbox[2]],
                                        [-bbox[0], bbox[1], bbox[2]],
                                        [-bbox[0], -bbox[1], bbox[2]],
                                        [-bbox[0], bbox[1], -bbox[2]],
                                        [-bbox[0], -bbox[1], -bbox[2]]])

        bbox_corners_world = self._transform_list_of_points(bbox_corners_object, pos, quat)
        bbox_corners_target = self._transform_list_of_points(bbox_corners_world, pos_target, Quaternion(), inv=True)
        min_id = np.argmin(np.linalg.norm(bbox_corners_target, axis=1))

        principal_corners_object = self._get_the_other_3_principal_corners(bbox_corners_object[min_id])
        principal_corners_world = self._transform_list_of_points(principal_corners_object, pos, quat)
        # Mujoco due to soft contacts has bottom under 0. Enforce to zero
        principal_corners_world[:, 2][principal_corners_world[:, 2] < 0] = 0

        if self.translate_wrt_target:
            principal_corners_target = self._transform_list_of_points(principal_corners_world,
                                                                      np.append(pos_target[0:2], 0), Quaternion(),
                                                                      inv=True)
            return principal_corners_target

        return principal_corners_world

    def _get_principal_corner_target(self):
        pos = self.poses[0, 0:3]
        quat = Quaternion(self.poses[0, 3], self.poses[0, 4], self.poses[0, 5], self.poses[0, 6])
        bbox = self.bounding_box[0]
        bbox[2] = -bbox[2]

        principal_corners_object = self._get_the_other_3_principal_corners(bbox)
        principal_corners_world = self._transform_list_of_points(principal_corners_object, pos=pos, quat=quat)
        # Mujoco due to soft contacts has bottom under 0. Enforce to zero
        principal_corners_world[:, 2][principal_corners_world[:, 2] < 0] = 0
        if self.translate_wrt_target:
            principal_corners_target = self._transform_list_of_points(principal_corners_world,
                                                                      pos=np.append(self.poses[0, 0:2], 0),
                                                                      quat=Quaternion(), inv=True)
            return principal_corners_target
        return principal_corners_world

    def _transform_list_of_points(self, points, pos, quat, inv=False):
        assert points.shape[1] == 3
        matrix = np.eye(4)
        matrix[0:3, 3] = pos
        matrix[0:3, 0:3] = quat.rotation_matrix()
        if inv:
            matrix = np.linalg.inv(matrix)

        transformed_points = np.transpose(np.matmul(matrix, np.transpose(
            np.concatenate((points, np.ones((points.shape[0], 1))), axis=1))))[:, :3]
        return transformed_points

    def _get_the_other_3_principal_corners(self, first_corner):
        principal_corners = np.zeros((4, 3))
        x = first_corner[0]; y = first_corner[1]; z = first_corner[2]
        principal_corners[0] = np.array([x, y, z])
        principal_corners[1] = np.array([-x, y, z])
        principal_corners[2] = np.array([x, -y, z])
        principal_corners[3] = np.array([x, y, -z])
        return principal_corners

    def sort(self):
        if self.coordinates == 'cartesian':
            self.principal_corners[1:] = self.principal_corners[1:][np.argsort(np.linalg.norm(self.principal_corners[1:], axis=2)[:, 0])]
        elif self.coordinates == 'spherical':
            self.principal_corners[1:] = self.principal_corners[1:][np.argsort(self.principal_corners[1:, 0, 0])]

    def normalize(self):
        eps = 0.05
        if self.coordinates == 'cartesian':
            range_ = [[-self.surface_size[0] - eps, self.surface_size[0] + eps],
                      [-self.surface_size[1] - eps, self.surface_size[1] + eps],
                      [0, 0.2]]
            max_surface_range = [[-2 * np.linalg.norm(self.surface_size) + eps,
                                  2 * np.linalg.norm(self.surface_size) + eps],
                                 [-2 * np.linalg.norm(self.surface_size) + eps,
                                  2 * np.linalg.norm(self.surface_size) + eps]]
        elif self.coordinates == 'spherical':
            range_ = [[0, np.linalg.norm(self.surface_size) + eps],
                      [-np.pi, np.pi],
                      [-np.pi, np.pi]]
            max_surface_range = [[0, 2 * np.linalg.norm(self.surface_size) + eps],
                                 [-np.pi, np.pi]]

        for i in range(3):
            self.principal_corners[:, :, i] = min_max_scale(self.principal_corners[:, :, i],
                                                            range=range_[i],
                                                            target_range=self.range_norm)

        for i in range(2):
            self.surface_edges[:, i] = min_max_scale(self.surface_edges[:, i], range=max_surface_range[i],
                                                     target_range=self.range_norm)

    def array(self):
        array = np.concatenate((self.principal_corners, self.range_norm[0] * np.ones(
            (int(self.max_n_objects - self.principal_corners.shape[0]), 4, 3)))).flatten()
        array = np.append(array, self.surface_distances.flatten())
        array = np.append(array, self.init_distance_from_target)

        # assert (array <= 1).all()
        # assert (array >= -1).all()

        # assert array.shape[0] == self.dim()
        return array

    def rotate(self, angle):
        for i in range(self.n_objects):
            self.principal_corners[i] = self._transform_list_of_points(self.principal_corners[i], pos=np.zeros(3),
                                                                       quat=Quaternion.from_rotation_matrix(
                                                                           rot_z(angle)))

    def plot(self, centroids=False, action=None, ax=None):
        from mpl_toolkits.mplot3d import Axes3D
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)

        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.principal_corners_plot.shape[0])))

        centroids_ = np.mean(self.principal_corners_plot, axis=1)
        for object in range(self.principal_corners_plot.shape[0]):
            c = next(color)
            for corner in range(self.principal_corners_plot.shape[1]):
                ax.plot([self.principal_corners_plot[object, corner, 0]],
                        [self.principal_corners_plot[object, corner, 1]],
                        [self.principal_corners_plot[object, corner, 2]], color=c, marker='o')

                if corner > 0:
                    ax.plot([self.principal_corners_plot[object, 0, 0], self.principal_corners_plot[object, corner, 0]],
                            [self.principal_corners_plot[object, 0, 1], self.principal_corners_plot[object, corner, 1]],
                            [self.principal_corners_plot[object, 0, 2], self.principal_corners_plot[object, corner, 2]],
                            color=c, linestyle='-')

                if centroids:
                    ax.plot([centroids_[object, 0]],
                            [centroids_[object, 1]],
                            [centroids_[object, 2]], color=c, marker='o')

        if action is not None:
            init = action[3] * np.array([cos(action[1]), sin(action[1])])
            final = -action[2] * np.array([cos(action[1]), sin(action[1])])
            # init += self.poses[0, 0:2]
            # final += self.poses[0, 0:2]
            ax.plot([init[0]], [init[1]], [0], color=[0, 0, 0], marker='o')
            ax.plot([final[0]], [final[1]], [0], color=[1, 1, 1], marker='o')
            ax.plot([init[0], final[0]], [init[1], final[1]], [0, 0], color=[0, 0, 0], linestyle='-')

        # ax.axis([-0.25, 0.25, -0.25, 0.25])
        ax.axis('equal')
        # plt.show()

    @staticmethod
    def dim():
        return 14 * 4 * 3 + 4 + 1 # TODO: hardcoded max n objects


def get_actor_visual_feature(heightmap, mask, target_bounding_box_z, finger_height, angle=0, primitive=0, plot=False,
                             maskout_target=False, crop_area=[128, 128], pooling=True, singulation_distance=0.03,
                             pixels_to_m=0.0012):
    heightmap_ = heightmap.copy()
    """Angle in deg"""
    # Calculate fused visual feature
    thresholded = np.zeros(heightmap.shape)
    if primitive == 0:
        threshold = 0  # assuming that the env doesnt spawn flat
    elif primitive == 1:
        threshold = 2 * target_bounding_box_z + 1.1 * finger_height
        masks = np.argwhere(mask == 255)
        if masks.size != 0:
            center = np.ones((masks.shape[0], 2))
            center[:, 0] *= mask.shape[0] / 2
            center[:, 1] *= mask.shape[1] / 2
            max_dist = np.max(np.linalg.norm(masks - center, axis=1))
            singulation_distance_in_pxl = int(np.ceil(singulation_distance / pixels_to_m)) + max_dist
            circle_mask = cv_tools.get_circle_mask(heightmap.shape[0], heightmap.shape[1],
                                                   singulation_distance_in_pxl, inverse=True)
            heightmap_ = cv_tools.Feature(heightmap_).mask_out(circle_mask).array()
    else:
        raise ValueError()

    if threshold < 0:
        threshold = 0
    thresholded[heightmap_ > threshold] = 1
    thresholded[mask > 0] = 0.5
    visual_feature = cv_tools.Feature(thresholded)
    if maskout_target:
        visual_feature = visual_feature.mask_out(mask)
    visual_feature = visual_feature.rotate(angle)
    visual_feature = visual_feature.crop(crop_area[0], crop_area[1])
    if pooling:
        visual_feature = visual_feature.pooling(kernel=[2, 2], stride=2, mode='AVG')
    if plot:
        visual_feature.plot()
    return visual_feature.array()


def get_asymmetric_actor_feature(autoencoder, normalizer, heightmap, mask, target_bounding_box_z, finger_height,
                                 target_pos, surface_distances,
                                 angle=0, primitive=0, plot=False):
    """Angle in rad"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    """Angle in rad"""
    angle_deg = angle * (180 / np.pi)
    visual_feature = get_actor_visual_feature(heightmap, mask, target_bounding_box_z, finger_height, angle_deg,
                                              primitive, plot=False)
    visual_feature = torch.FloatTensor(visual_feature).reshape(1, 1, visual_feature.shape[0],
                                                               visual_feature.shape[1]).to('cpu')
    # start.record()
    # with torch.no_grad():
        # print(autoencoder.encoder.conv4(autoencoder.encoder.conv3(autoencoder.encoder.conv2(autoencoder.encoder.conv1(visual_feature)))))
    latent = autoencoder.encoder(visual_feature)

    # end.record()
    # torch.cuda.synchronize()
    # print('cuda time', start.elapsed_time(end))
    # print('time:', time.time() - start)
    normalized_latent = latent.detach().cpu().numpy()
    if normalizer is not None:
        normalized_latent = normalizer.transform(latent.detach().cpu().numpy())

    # surface_edges_ = surface_edges - target_pos
    # rot_2d = np.array([[cos(angle), -sin(angle)],
    #                    [sin(angle), cos(angle)]])
    # surface_edges_ = np.transpose(np.matmul(rot_2d, np.transpose(surface_edges_)))

    if plot:
        ae_output = autoencoder(visual_feature).detach().cpu().numpy()[0, 0, :, :]
        visual_feature = visual_feature.detach().cpu().numpy()[0, 0, :, :]
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(visual_feature, cmap='gray', vmin=np.min(visual_feature), vmax=np.max(visual_feature))
        ax[1].imshow(ae_output, cmap='gray', vmin=np.min(ae_output), vmax=np.max(ae_output))
        plt.show()

    # if primitive == 1:
    #     return np.append(normalized_latent, np.zeros(surface_distances.flatten().shape))

    return np.append(normalized_latent, surface_distances.flatten())


def get_asymmetric_actor_feature_from_dict(obs_dict, autoencoder, normalizer, angle=0, primitive=0, plot=False):
    heightmap = obs_dict['heightmap_mask'][0]
    mask = obs_dict['heightmap_mask'][1]
    target_bounding_box_z = obs_dict['target_bounding_box'][2]
    finger_height = obs_dict['finger_height']
    surface_edges = obs_dict['surface_edges']

    if obs_dict['walls']:
        surface_distances = get_distances_from_walls(obs_dict)
    else:
        surface_distances = [obs_dict['surface_size'][0] - obs_dict['object_poses'][0, 0], \
                             obs_dict['surface_size'][0] + obs_dict['object_poses'][0, 0], \
                             obs_dict['surface_size'][1] - obs_dict['object_poses'][0, 1], \
                             obs_dict['surface_size'][1] + obs_dict['object_poses'][0, 1]]

    surface_distances = np.array([x / 0.5 for x in surface_distances])

    target_pos = obs_dict['object_poses'][0, 0:2]
    return get_asymmetric_actor_feature(autoencoder, normalizer, heightmap, mask, target_bounding_box_z, finger_height,
                                        target_pos, surface_distances, angle, primitive, plot)

def detect_singulation_from_actor_visual_feature(obs_dict):
    heightmap = obs_dict['heightmap_mask'][0]
    mask = obs_dict['heightmap_mask'][1]
    target_bounding_box_z = obs_dict['target_bounding_box'][2]
    finger_height = obs_dict['finger_height']
    singulation_distance = obs_dict['singulation_distance'][0]
    pixels_to_m = obs_dict['pixels_to_m'][0]
    singulation_distance_in_pxl = int(np.ceil(singulation_distance / pixels_to_m))
    visual_feature = get_actor_visual_feature(heightmap, mask, target_bounding_box_z, finger_height, angle=0,
                                              primitive=0, plot=False, maskout_target=True)
    singulation_feature = cv_tools.Feature(visual_feature).crop(singulation_distance_in_pxl,
                                                                singulation_distance_in_pxl)
    if np.max(singulation_feature.array()) < 1.0:
        return True
    return False


def detect_singulation_from_real_state(obs_dict, singulate_from_walls=False):
    # Returns the minimum distance of target from the obstacles'''
    n_objects = int(obs_dict['n_objects'])
    target_pose = obs_dict['object_poses'][0]
    target_bbox = obs_dict['object_bounding_box'][0]

    distances = 100 * np.ones((int(n_objects),))
    for i in range(1, n_objects):
        obstacle_pose = obs_dict['object_poses'][i]
        obstacle_bbox = obs_dict['object_bounding_box'][i]

        if obs_dict['object_above_table'][i]:
            distances[i] = get_distance_of_two_bbox(target_pose, target_bbox, obstacle_pose, obstacle_bbox)

    min_distance = np.min(distances)

    singulation_distance = obs_dict['singulation_distance'][0]

    singulation_condition = min_distance > singulation_distance

    if singulate_from_walls:
        fixed_distances = get_distances_from_walls(obs_dict)
        min_fixed_distance = np.min(fixed_distances)
        singulation_condition = min_distance > singulation_distance and min_fixed_distance > singulation_distance

    if singulation_condition:
        return True

    return False

def detect_target_stuck_on_wall(obs_dict):
    target_pose = obs_dict['object_poses'][0]

    if np.min(get_distances_from_walls(obs_dict)) > 0.001:
        return False

    distances = get_distances_from_walls(obs_dict)
    argmin = np.argmin(distances)
    target_x_axis = Quaternion.from_vector(target_pose[3:]).rotation_matrix()[:, 0]
    wall_pose = obs_dict['fixed_object_poses'][argmin]
    wall_x_axis = Quaternion.from_vector(wall_pose[3:]).rotation_matrix()[:, 0]
    dot_pr = np.abs(np.dot(target_x_axis, wall_x_axis))

    if dot_pr > 0.80 or dot_pr < 0.2:
        return True

    return False

def detect_empty_action_from_real_state(obs_dict, obs_dict_prev):
    '''Its not the best detection because if an object is moved not because of the finger it will return a non-empty push'''
    poses = obs_dict['object_poses'][obs_dict['object_above_table']][:, 0:3]
    poses_prev = obs_dict_prev['object_poses'][obs_dict_prev['object_above_table']][:, 0:3]
    if len(poses) == len(poses_prev):
        error = np.abs(poses - poses_prev)
        return (error < 1e-3).all()
    return False

def get_distances_from_walls(obs_dict):
    '''Returns the 4 distances of the target from the walls'''
    n_fixed_objects = int(obs_dict['n_fixed_objects'])
    target_pose = obs_dict['object_poses'][0]
    target_bbox = obs_dict['object_bounding_box'][0]

    fixed_distances = 100 * np.ones((int(n_fixed_objects),))
    for i in range(0, n_fixed_objects):
        fixed_obstacle_pose = obs_dict['fixed_object_poses'][i]
        fixed_obstacle_bbox = obs_dict['fixed_object_bounding_box'][i]
        fixed_distances[i] = get_distance_of_two_bbox(target_pose, target_bbox, fixed_obstacle_pose,
                                                      fixed_obstacle_bbox, plot=False)

    return fixed_distances


def get_object_height(pose, bounding_box):
    bbox = bounding_box
    bbox_corners_object = np.array([[bbox[0], bbox[1], bbox[2]],
                                    [bbox[0], -bbox[1], bbox[2]],
                                    [bbox[0], bbox[1], -bbox[2]],
                                    [bbox[0], -bbox[1], -bbox[2]],
                                    [-bbox[0], bbox[1], bbox[2]],
                                    [-bbox[0], -bbox[1], bbox[2]],
                                    [-bbox[0], bbox[1], -bbox[2]],
                                    [-bbox[0], -bbox[1], -bbox[2]]])
    pos = pose[0:3]
    quat = Quaternion(pose[3], pose[4], pose[5], pose[6])
    corners = transform_list_of_points(bbox_corners_object, pos, quat)
    dists = np.zeros(corners.shape[0])
    normal = np.zeros(3)
    normal[2] = 1
    for i in range(corners.shape[0]):
        v = corners[i] - np.zeros(3)
        dists[i] = abs(np.dot(v, normal))
    return np.max(dists)



def push_obstacle_feature_includes_affordances(obs_dict):
    heightmap = obs_dict['heightmap_mask'][0]
    mask = obs_dict['heightmap_mask'][1]
    target_bounding_box_z = obs_dict['target_bounding_box'][2]
    finger_height = obs_dict['finger_height']
    pixels_to_m = obs_dict['pixels_to_m'][0]
    push_distance = obs_dict['push_distance_range'][1]
    singulation_distance_in_pxl = int(np.ceil(push_distance / pixels_to_m))
    crop_area = [singulation_distance_in_pxl, singulation_distance_in_pxl]

    visual_feature = get_actor_visual_feature(heightmap, mask, target_bounding_box_z, finger_height, angle=0,
                                              primitive=1, plot=False, maskout_target=True, crop_area=crop_area,
                                              pooling=False)

    circle_mask = cv_tools.get_circle_mask(visual_feature.shape[0], visual_feature.shape[1],
                                           singulation_distance_in_pxl, inverse=True)
    visual_feature = cv_tools.Feature(visual_feature).mask_out(circle_mask).array()
    if np.max(visual_feature) < 1.0:
        return False
    return True

def slide_is_eligible(obs_dict):
    distances = get_distances_from_walls(obs_dict)
    if np.min(distances) < 0.03:
        return True
    return False

def get_valid_primitives(obs_dict, n_primitives=3):
    valid_primitives = np.zeros(n_primitives, dtype=np.bool)

    valid_primitives[0] = True

    if push_obstacle_feature_includes_affordances(obs_dict):
        valid_primitives[1] = True


    if n_primitives >=3:
        if slide_is_eligible(obs_dict):
            valid_primitives[2] = True

    return valid_primitives, np.argwhere(valid_primitives == True).flatten()


class PrimitiveFeature:
    def __init__(self, heightmap, mask, angle=0, name=None):
        self.name = name
        self.angle = angle
        self.heightmap = Feature(heightmap).rotate(angle)
        self.mask = Feature(mask).rotate(angle)
        self.visual_feature = self.get_visual_feature()

    def get_visual_feature(self):
        raise NotImplementedError()

    def array(self):
        return self.visual_feature.flatten()

    def plot(self):
        self.visual_feature.plot()


class PushTargetFeature(PrimitiveFeature):
    def __init__(self, obs_dict, angle=0):
        self.target_bounding_box_z = obs_dict['target_bounding_box'][2]
        self.finger_height = obs_dict['finger_height']
        self.crop_area = obs_dict['observation_area'].astype(np.int32)
        self.target_pos = obs_dict['target_pos'][:2]
        self.surface_size = obs_dict['surface_size'][:2]
        heightmap = obs_dict['heightmap_mask'][0, :]
        mask = obs_dict['heightmap_mask'][1, :]
        super().__init__(heightmap, mask, angle, name='PushTarget')

    def get_visual_feature(self):
        thresholded = np.zeros(self.heightmap.array().shape)
        threshold = self.target_bounding_box_z - 1.5 * self.finger_height
        if threshold < 0:
            threshold = 0
        thresholded[self.heightmap.array() > threshold] = 1
        thresholded[self.mask.array() > 0] = -1
        feature = Feature(thresholded)
        # feature = feature.crop(self.crop_area[0], self.crop_area[1])
        # feature = feature.pooling(mode='AVG')
        return feature

    def array(self):
        array = super().array()
        target_observation_ratio = TargetObjectConvexHull(self.mask.array()).area() / (
                    self.crop_area[0] * self.crop_area[1])
        array = np.append(array, target_observation_ratio)

        theta = min_max_scale(self.angle, range=[-180, 180], target_range=[-pi, pi])
        rot_2d = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

        max_surface_edge = np.linalg.norm(self.surface_size)

        target_pos = np.matmul(rot_2d, self.target_pos)
        target_pos = min_max_scale(target_pos, range=[-max_surface_edge, max_surface_edge], target_range=[-1, 1])
        surface_edge = np.matmul(rot_2d, self.surface_size)
        surface_edge = min_max_scale(surface_edge, range=[-max_surface_edge, max_surface_edge], target_range=[-1, 1])

        array = np.append(array, target_pos)
        array = np.append(array, surface_edge)

        assert (array <= 1).all()
        assert (array >= -1).all()

        assert array.shape[0] == self.dim()
        return array

    def plot(self):
        self.visual_feature.plot()

        print()

    @staticmethod
    def dim():
        return 630


class PushObstacleFeature(PrimitiveFeature):
    def __init__(self, obs_dict, angle=0):
        self.target_bounding_box_z = obs_dict['target_bounding_box'][2]
        self.finger_height = obs_dict['finger_height']
        self.crop_area = obs_dict['max_singulation_area'].astype(np.int32)
        self.target_distances_from_limits = obs_dict['target_distances_from_limits']
        heightmap = obs_dict['heightmap_mask'][0, :]
        mask = obs_dict['heightmap_mask'][1, :]
        super().__init__(heightmap, mask, angle, name='PushTarget')

    def get_visual_feature(self):
        thresholded = np.zeros(self.heightmap.array().shape)
        threshold = 2 * self.target_bounding_box_z + 1.5 * self.finger_height
        thresholded[self.heightmap.array() > threshold] = 1
        thresholded[self.mask.array() > 0] = -1
        feature = Feature(thresholded)
        feature = feature.crop(self.crop_area[0], self.crop_area[1])
        feature = feature.pooling()
        return feature

    def array(self):
        array = super().array()
        target_observation_ratio = TargetObjectConvexHull(self.mask.array()).area() / (self.crop_area[0] * self.crop_area[1])
        array = np.append(array, target_observation_ratio)
        assert array.shape[0] == self.dim()
        return array

    @staticmethod
    def dim():
        return 401


class GraspTargetFeature(PrimitiveFeature):
    def __init__(self, obs_dict, angle=0):
        self.target_bounding_box_z = obs_dict['target_bounding_box'][2]
        self.finger_height = obs_dict['finger_height']
        self.crop_area = [40, 40]
        self.target_distances_from_limits = obs_dict['target_distances_from_limits']
        heightmap = obs_dict['heightmap_mask'][0, :]
        mask = obs_dict['heightmap_mask'][1, :]
        super().__init__(heightmap, mask, angle, name='GraspTarget')

    def get_visual_feature(self):
        thresholded = np.zeros(self.heightmap.array().shape)
        threshold = self.target_bounding_box_z - 1.5 * self.finger_height
        if threshold < 0:
            threshold = 0
        thresholded[self.heightmap.array() > threshold] = 1
        thresholded[self.mask.array() > 0] = -1
        feature = Feature(thresholded)
        feature = feature.crop(self.crop_area[0], self.crop_area[1])
        feature = feature.pooling()
        return feature

    def array(self):
        array = super().array()
        target_observation_ratio = TargetObjectConvexHull(self.mask.array()).area() / (self.crop_area[0] * self.crop_area[1])
        array = np.append(array, target_observation_ratio)
        assert array.shape[0] == self.dim()
        return array

    @staticmethod
    def dim():
        return 401

def get_icra_feature(obs_dict, rotations=8):
    debug('get_icra_feature: Start')
    point_cloud = cv_tools.PointCloud()
    point_cloud.points = obs_dict['point_cloud']
    debug('object_poses:', obs_dict['object_poses'])
    heightmaps = point_cloud.generate_height_map(rotations=rotations)
    rot_angle = 360 / rotations
    features = []
    for i in range(0, len(heightmaps)):
        feature = cv_tools.Feature(heightmaps[i])
        feature = feature.crop(32, 32)
        feature = feature.pooling()
        feature = feature.normalize(0.06).flatten()
        feature = np.append(feature, i * rot_angle)
        feature = np.append(feature, obs_dict['object_bounding_box'][0][:2] / 0.03)

        # Add the distance of the object from the edge
        distances = [obs_dict['surface_size'][0] - obs_dict['object_poses'][0, 0], \
                     obs_dict['surface_size'][0] + obs_dict['object_poses'][0, 0], \
                     obs_dict['surface_size'][1] - obs_dict['object_poses'][0, 1], \
                     obs_dict['surface_size'][1] + obs_dict['object_poses'][0, 1]]

        distances = [x / 0.5 for x in distances]

        feature = np.append(feature, distances)
        features.append(feature)
    final_feature = np.append(features[0], features[1], axis=0)
    for i in range(2, len(features)):
        final_feature = np.append(final_feature, features[i], axis=0)
    debug('get_icra_feature: End')
    return final_feature


# Action: Pushing Primitive Actions
# ---------------------------------

class PushAction:
    """
    A pushing action of two 3D points for init and final pos.
    """

    def __init__(self, p1, p2):
        self.p1 = p1.copy()
        self.p2 = p2.copy()

    def get_init_pos(self):
        return self.p1

    def get_final_pos(self):
        return self.p2

    def get_duration(self, distance_per_sec=0.1):
        return np.linalg.norm(self.get_init_pos() - self.get_final_pos()) / distance_per_sec

    def translate(self, p):
        self.p1 += p
        self.p2 += p
        return self

    def rotate(self, rot):
        """
        Rot: rotation matrix
        """
        self.p1 = np.matmul(rot, self.p1)
        self.p2 = np.matmul(rot, self.p2)
        return self

    def plot(self):
        raise NotImplementedError()

    def array(self):
        raise NotImplementedError()


class PushAction2D(PushAction):
    def __init__(self, p1, p2, z=0):
        super(PushAction2D, self).__init__(np.append(p1, z), np.append(p2, z))

    def translate(self, p):
        return super(PushAction2D, self).translate(np.append(p, 0))

    def rotate(self, angle):
        """
        Angle in rad.
        """
        return super(PushAction2D, self).rotate(rot_z(angle))

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        color = [0, 0, 0]
        ax.plot(self.p1[0], self.p1[1], color=color, marker='o')
        ax.plot(self.p2[0], self.p2[1], color=color, marker='.')
        ax.plot([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]], color=color, linestyle='-')
        return self


class PushTarget2D(PushAction2D):
    """
    A basic push target push with the push distance (for having the info where the target is)
    """

    def __init__(self, p1, p2, z, push_distance):
        self.push_distance = push_distance
        super(PushTarget2D, self).__init__(p1, p2, z)

    def get_target(self):
        return self.p2 - ((self.p2 - self.p1) / np.linalg.norm(self.p2 - self.p1)) * self.push_distance

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        color = [1, 0, 0]
        target = self.get_target()
        ax.plot(target[0], target[1], color=color, marker='o')

        return super(PushTarget2D, self).plot(ax)


class PushTargetWithObstacleAvoidance(PushTarget2D):
    """
    A 2D push for pushing target which uses the 2D convex hull of the object to enforce obstacle avoidance.
    convex_hull: A list of Linesegments2D. Should by in order cyclic, in order to calculate the centroid correctly
    Theta, push_distance, distance assumed to be in [-1, 1]
    """

    def __init__(self, theta, push_distance, distance, push_distance_range, init_distance_range, convex_hull,
                 object_height, finger_size):
        self.convex_hull = convex_hull  # store for plotting purposes

        # Calculate the centroid from the convex hull
        # -------------------------------------------
        hull_points = np.zeros((len(convex_hull), 2))
        for i in range(len(convex_hull)):
            hull_points[i] = convex_hull[i].p1
        centroid = get_centroid_convex_hull(hull_points)

        theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-pi, pi])

        # Calculate the initial point p1 from convex hull
        # -----------------------------------------------
        # Calculate the intersection point between the direction of the
        # push theta and the convex hull (four line segments)
        direction = np.array([cos(theta_), sin(theta_)])
        line_segment = LineSegment2D(centroid, centroid + 10 * direction)
        min_point = line_segment.get_first_intersection_point(convex_hull)
        min_point += (finger_size + 0.008) * direction
        max_point = centroid + (np.linalg.norm(centroid - min_point) + init_distance_range[1]) * direction
        distance_line_segment = LineSegment2D(min_point, max_point)
        lambd = min_max_scale(distance, range=[-1, 1], target_range=[0, 1])
        p1 = distance_line_segment.get_point(lambd)

        # Calculate the initial point p2
        # ------------------------------
        direction = np.array([cos(theta_ + pi), sin(theta_ + pi)])
        min_point = centroid
        max_point = centroid + push_distance_range[1] * direction
        push_line_segment = LineSegment2D(min_point, max_point)
        lambd = min_max_scale(push_distance, range=[-1, 1], target_range=[0, 1])
        p2 = push_line_segment.get_point(lambd)

        # Calculate height (z) of the push
        # --------------------------------
        if object_height - finger_size > 0:
            offset = object_height - finger_size
        else:
            offset = 0
        z = float(finger_size + offset + 0.001)

        super(PushTargetWithObstacleAvoidance, self).__init__(p1, p2, z, np.linalg.norm(p2 - centroid))

    def translate(self, p):
        for i in range(len(self.convex_hull)):
            self.convex_hull[i] = self.convex_hull[i].translate(p)

        return super(PushTargetWithObstacleAvoidance, self).translate(p)

    def rotate(self, angle):
        for i in range(len(self.convex_hull)):
            self.convex_hull[i] = self.convex_hull[i].rotate(angle)

        return super(PushTargetWithObstacleAvoidance, self).rotate(angle)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        color = [0, 0, 1]
        target = self.get_target()
        ax.plot(target[0], target[1], color=color, marker='o')

        for line_segment in self.convex_hull:
            ax.plot(line_segment.p1[0], line_segment.p1[1], color=color, marker='o')
            ax.plot(line_segment.p2[0], line_segment.p2[1], color=color, marker='.')
            ax.plot([line_segment.p1[0], line_segment.p2[0]], [line_segment.p1[1], line_segment.p2[1]],
                    color=color, linestyle='-')

        return super(PushTargetWithObstacleAvoidance, self).plot(ax)


class PushTargetRealWithObstacleAvoidance(PushTargetWithObstacleAvoidance):
    def __init__(self, obs_dict, theta, push_distance, distance, push_distance_range, init_distance_range,
                 translate_wrt_target=True):
        # Store the ranges to use it in case array is called with normalized
        self.push_distance_range = push_distance_range
        self.init_distance_range = init_distance_range
        # Get the vertices of the bounding box from real state, calculate the fourth vertice and this is the convex hull
        target_principal_corners = RealState(obs_dict, sort=False, normalize=False,
                                             spherical=False,
                                             translate_wrt_target=translate_wrt_target).principal_corners[0]

        fourth_point = target_principal_corners[2][0:2] - target_principal_corners[0][0:2]
        fourth_point += target_principal_corners[1][0:2]
        convex_hull = [LineSegment2D(target_principal_corners[0][0:2], target_principal_corners[1][0:2]),
                       LineSegment2D(target_principal_corners[1][0:2], fourth_point),
                       LineSegment2D(fourth_point, target_principal_corners[2][0:2]),
                       LineSegment2D(target_principal_corners[2][0:2], target_principal_corners[0][0:2])]

        finger_size = obs_dict['finger_height']
        object_height = target_principal_corners[3][2] / 2

        super(PushTargetRealWithObstacleAvoidance, self).__init__(theta=theta, push_distance=push_distance,
                                                                  distance=distance,
                                                                  push_distance_range=push_distance_range,
                                                                  init_distance_range=init_distance_range,
                                                                  convex_hull=convex_hull, object_height=object_height,
                                                                  finger_size=finger_size)

    def array(self):
        x_ = min_max_scale(self.p1[0], range=[-self.init_distance_range[1], self.init_distance_range[1]],
                           target_range=[-1, 1])
        y_ = min_max_scale(self.p1[1], range=[-self.init_distance_range[1], self.init_distance_range[1]],
                           target_range=[-1, 1])
        push_distance = np.linalg.norm(self.centroid - self.p2)
        push_distance = min_max_scale(push_distance, range=self.push_distance_range, target_range=[-1, 1])

        return np.array([0, x_, y_, push_distance])


class PushTargetReal(PushTarget2D):
    def __init__(self, theta, push_distance, distance, push_distance_range, init_distance_range, object_height,
                 finger_size):
        self.push_distance_range = push_distance_range
        self.init_distance_range = init_distance_range
        theta = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
        r = min_max_scale(distance, range=[-1, 1], target_range=init_distance_range)
        push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=push_distance_range)
        p1 = np.array([r * np.cos(theta), r * np.sin(theta)])
        p2 = - push_distance_ * np.array([cos(theta), sin(theta)])

        # Calculate height (z) of the push
        # --------------------------------
        if object_height - finger_size > 0:
            offset = object_height - finger_size
        else:
            offset = 0
        z = float(finger_size + offset + 0.001)

        super(PushTargetReal, self).__init__(p1, p2, z, push_distance_)

    def array(self):
        x_ = min_max_scale(self.p1[0], range=self.workspace, target_range=[-1, 1])
        y_ = min_max_scale(self.p1[1], range=self.workspace, target_range=[-1, 1])
        push_distance = min_max_scale(self.push_distance, range=self.push_distance_range, target_range=[-1, 1])
        return np.array([0, x_, y_, push_distance])


class PushTargetRealCartesian(PushTarget2D):
    '''Init pos if target is at zero and push distance. Then you can translate the push.'''

    def __init__(self, x_init, y_init, push_distance, object_height, finger_size):
        x_ = x_init
        y_ = y_init
        push_distance_ = push_distance
        p1 = np.array([x_, y_])
        theta = np.arctan2(y_, x_)
        p2 = - push_distance_ * np.array([cos(theta), sin(theta)])

        # Calculate height (z) of the push
        # --------------------------------
        if object_height - finger_size > 0:
            offset = object_height - finger_size
        else:
            offset = 0
        z = float(finger_size + offset + 0.001)

        super(PushTargetRealCartesian, self).__init__(p1, p2, z, push_distance_)


class PushTargetRealCartesianNormalized(PushTarget2D):
    '''Init pos if target is at zero and push distance. Then you can translate the push.'''

    def __init__(self, x_init, y_init, push_distance, push_distance_range, max_init_distance, object_height,
                 finger_size):
        self.push_distance_range = push_distance_range
        self.workspace = [-max_init_distance, max_init_distance]
        x_ = min_max_scale(x_init, range=[-1, 1], target_range=self.workspace)
        y_ = min_max_scale(y_init, range=[-1, 1], target_range=self.workspace)
        push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=push_distance_range)
        super(PushTargetRealCartesianNormalized, self).__init__(x_, y_, push_distance_, object_height, finger_size)

    def array(self):
        x_ = min_max_scale(self.p1[0], range=self.workspace, target_range=[-1, 1])
        y_ = min_max_scale(self.p1[1], range=self.workspace, target_range=[-1, 1])
        push_distance = min_max_scale(self.push_distance, range=self.push_distance_range, target_range=[-1, 1])
        return np.array([0, x_, y_, push_distance])


class PushTargetRealObjectAvoidance(PushTargetRealCartesian):
    def __init__(self, obs_dict, angle, push_distance, push_distance_range, init_distance_range, target_height,
                 finger_size, eps=0.05):
        angle_ = min_max_scale(angle, range=[-1, 1], target_range=[-np.pi, np.pi])
        push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=push_distance_range)
        eps_ = min_max_scale(eps, range=[-1, 1], target_range=[-np.pi, np.pi])

        max_init_distance = init_distance_range[1]
        x_init = max_init_distance * np.cos(angle_)
        y_init = max_init_distance * np.sin(angle_)
        preprocessed = preprocess_real_state(obs_dict, max_init_distance, 0, primitive=0)

        point_cloud = get_table_point_cloud(preprocessed['object_poses'][preprocessed['object_above_table']],
                                            preprocessed['object_bounding_box'][preprocessed['object_above_table']],
                                            workspace=[max_init_distance, max_init_distance],
                                            density=128)

        point_cloud_cylindrical = np.zeros(point_cloud.shape)
        point_cloud_cylindrical[:, 0] = np.sqrt(point_cloud[:, 0] ** 2 + point_cloud[:, 1] ** 2)
        point_cloud_cylindrical[:, 1] = np.arctan2(point_cloud[:, 1], point_cloud[:, 0])

        error = np.abs(angle_ - point_cloud_cylindrical[:, 1]) < eps_
        point_cloud_cylindrical = point_cloud_cylindrical[error]
        arr = point_cloud_cylindrical[:, 0]
        if arr.size != 0:
            result = np.where(arr == np.min(arr))[0]

            point_cloud_ = np.zeros(point_cloud_cylindrical.shape)
            point_cloud_[:, 0] = point_cloud_cylindrical[:, 0] * np.cos(point_cloud_cylindrical[:, 1])
            point_cloud_[:, 1] = point_cloud_cylindrical[:, 0] * np.sin(point_cloud_cylindrical[:, 1])
            minn = point_cloud_[result[0], :]

            x_init = minn[0]
            y_init = minn[1]

        push_avoid = PushTargetRealWithObstacleAvoidance(obs_dict, angle, push_distance=-1, distance=-1,
                                                         push_distance_range=[0, 0.1], init_distance_range=[0, 0.1])
        dist_from_target = np.linalg.norm(np.array([x_init, y_init]) - push_avoid.get_init_pos()[:2])
        self.init_distance_from_target = min_max_scale(dist_from_target, range=[0, max_init_distance],
                                                       target_range=[-1, 1])
        # Uncomment to plot
        # fig, ax = plt.subplots()
        # ax.scatter(point_cloud[:, 0], point_cloud[:, 1])
        # ax.scatter(point_cloud_[:10, 0], point_cloud_[:10, 1], color='r')
        # ax.scatter(minn[0], minn[1], color='b')
        # plt.show()
        super(PushTargetRealObjectAvoidance, self).__init__(x_init=x_init, y_init=y_init, push_distance=push_distance_,
                                                            object_height=target_height, finger_size=finger_size)


class PushTargetDepthObjectAvoidance(PushTargetRealCartesian):
    def __init__(self, obs_dict, angle, push_distance, push_distance_range, init_distance_range, target_height,
                 finger_length, finger_height, pixels_to_m, camera, rgb_to_camera_frame, camera_pose):
        angle_ = min_max_scale(angle, range=[-1, 1], target_range=[-np.pi, np.pi])
        push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=push_distance_range)

        max_init_distance = init_distance_range[1]
        x_init = max_init_distance * np.cos(angle_)
        y_init = max_init_distance * np.sin(angle_)

        raw_depth_ = obs_dict['raw_depth'].copy()
        table_depth = np.max(raw_depth_)

        # TODO: The patch has unit orientation and is the norm of the finger length. Optimally the patch should be
        # calculated using an oriented square.
        patch_size = 2 * int(ceil(np.linalg.norm([finger_length, finger_length]) / pixels_to_m))
        centroid_pxl = np.zeros(2, dtype=np.int32)
        centroid_pxl[0] = obs_dict['centroid_pxl'][1]
        centroid_pxl[1] = obs_dict['centroid_pxl'][0]
        r = 0
        step = 2
        while True:
            c = np.array([r * np.sin(-angle_), r * cos(-angle_)]).astype(np.int32)
            patch_center = centroid_pxl + c
            if patch_center[0] > raw_depth_.shape[0] or patch_center[1] > raw_depth_.shape[1]:
                break
            # calc patch position and extract the patch
            patch_x = int(patch_center[0] - patch_size / 2.)
            patch_y = int(patch_center[1] - patch_size / 2.)
            patch_image = raw_depth_[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]

            # fig, ax = plt.subplots(1)
            # ax.imshow(raw_depth_)
            # rect = patches.Rectangle((patch_y, patch_x), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
            # plt.show()
            # plt.imshow(patch_image)
            # plt.show()
            # print('abs', np.abs(patch_image - table_depth))
            # print('table', table_depth)

            if (np.abs(patch_image - table_depth) < 1e-3).all():
                z = raw_depth_[patch_center[0], patch_center[1]]
                patch_center_ = patch_center.copy()
                patch_center_[0] = patch_center[1]
                patch_center_[1] = patch_center[0]
                patch_center_image = camera.back_project(patch_center_, z)
                patch_center_camera = np.matmul(rgb_to_camera_frame, patch_center_image)
                # patch_center_camera = patch_center_image
                patch_center__ = np.matmul(camera_pose, np.array(
                    [patch_center_camera[0], patch_center_camera[1], 0, 1.0]))[:2]
                x_init_ = patch_center__[0] - obs_dict['object_poses'][0, 0]
                y_init_ = patch_center__[1] - obs_dict['object_poses'][0, 1]
                if np.linalg.norm([x_init_, y_init_]) < np.linalg.norm([x_init, y_init]):
                    x_init = x_init_
                    y_init = y_init_
                break
            r += step

        push_avoid = PushTargetRealWithObstacleAvoidance(obs_dict, angle, push_distance=-1, distance=-1,
                                                         push_distance_range=[0, 0.1], init_distance_range=[0, 0.1])
        dist_from_target = np.linalg.norm(np.array([x_init, y_init]) - push_avoid.get_init_pos()[:2])
        self.init_distance_from_target = min_max_scale(dist_from_target, range=[0, max_init_distance],
                                                       target_range=[-1, 1])
        super(PushTargetDepthObjectAvoidance, self).__init__(x_init=x_init, y_init=y_init, push_distance=push_distance_,
                                                             object_height=target_height, finger_size=finger_height)



class PushObstacle(PushAction2D):
    def __init__(self, theta, push_distance, push_distance_range, object_height, finger_height):
        self.push_distance_range = push_distance_range
        theta = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
        self.push_distance = min_max_scale(push_distance, range=[-1, 1], target_range=push_distance_range)
        p1 = np.zeros(2)
        p2 = self.push_distance * np.array([cos(theta), sin(theta)])

        # Calculate height (z) of the push
        # --------------------------------
        if object_height - finger_height > 0:
            offset = object_height - finger_height
        else:
            offset = 0
        z = float(finger_height + offset + 0.005)

        super(PushObstacle, self).__init__(p1, p2, z)

class PushObstacleICRA(PushObstacle):
    def __init__(self, action, nr_rotations, push_distance, push_distance_range, object_height, finger_height):
        theta = action * 2 * np.pi / nr_rotations
        debug('PushObstacleICRA: action:', action, 'nr_rotations:', nr_rotations, 'theta:', theta)
        if theta > np.pi:
            theta = -np.pi + abs(theta - np.pi)
            debug('PushObstacleICRA: action above pi changed to', theta)
        theta = min_max_scale(theta, range=[-np.pi, np.pi], target_range=[-1, 1])
        debug('PushObstacleICRA: final scaled theta:', theta)
        super(PushObstacleICRA, self).__init__(theta=theta, push_distance=push_distance,
                                               push_distance_range=push_distance_range, object_height=object_height,
                                               finger_height=finger_height)

class PushTargetICRA(PushTargetWithObstacleAvoidance):
    def __init__(self, action, nr_rotations, obs_dict, push_distance_range, translate_wrt_target=True):
        theta = action * 2 * np.pi / nr_rotations
        if theta > np.pi:
            theta = -np.pi + abs(theta - np.pi)
        theta = min_max_scale(theta, range=[-np.pi, np.pi], target_range=[-1, 1])

        self.push_distance_range = push_distance_range
        self.init_distance_range = [0, 0.1]
        # Get the vertices of the bounding box from real state, calculate the fourth vertice and this is the convex hull
        target_principal_corners = RealState(obs_dict, sort=False, normalize=False,
                                             spherical=False,
                                             translate_wrt_target=translate_wrt_target).principal_corners[0]

        fourth_point = target_principal_corners[2][0:2] - target_principal_corners[0][0:2]
        fourth_point += target_principal_corners[1][0:2]
        convex_hull = [LineSegment2D(target_principal_corners[0][0:2], target_principal_corners[1][0:2]),
                       LineSegment2D(target_principal_corners[1][0:2], fourth_point),
                       LineSegment2D(fourth_point, target_principal_corners[2][0:2]),
                       LineSegment2D(target_principal_corners[2][0:2], target_principal_corners[0][0:2])]

        angle, _ = rot2angleaxis(Quaternion.from_vector(obs_dict['object_poses'][0, 3:]).rotation_matrix())
        rotz = rot_z(angle)
        for i in range(4):
            convex_hull[i].p1 = np.matmul(rotz, np.append(convex_hull[i].p1, 0))[:2]
            convex_hull[i].p2 = np.matmul(rotz, np.append(convex_hull[i].p2, 0))[:2]

        finger_size = obs_dict['finger_height']
        object_height = target_principal_corners[3][2] / 2

        super(PushTargetICRA, self).__init__(theta=theta, push_distance=1,
                                                                  distance=-1,
                                                                  push_distance_range=push_distance_range,
                                                                  init_distance_range=[0, 0.1],
                                                                  convex_hull=convex_hull, object_height=object_height,
                                                                  finger_size=finger_size)

class PushExtraICRA(PushTarget2D):
    def __init__(self, action, nr_rotations, obs_dict):
        theta = action * 2 * np.pi / nr_rotations
        if theta > np.pi:
            theta = -np.pi + abs(theta - np.pi)

        r = np.linalg.norm(obs_dict['surface_size'])
        init_distance = np.linalg.norm(obs_dict['surface_size'])
        push_distance = 0.1
        p1 = init_distance * np.array([np.cos(theta), np.sin(theta)])
        p2 = - push_distance * np.array([cos(theta), sin(theta)])

        # Calculate height (z) of the push
        # --------------------------------
        finger_size = obs_dict['finger_height']
        object_height = obs_dict['object_bounding_box'][0, 2]
        if object_height - finger_size > 0:
            offset = object_height - finger_size
        else:
            offset = 0
        z = float(finger_size + offset + 0.001)

        super(PushExtraICRA, self).__init__(p1, p2, z, push_distance)



# Various utils methods
# ---------------------

class SingulationCondition:
    def __init__(self, finger_size, pixels_to_m, finger_max_spread, step=2, theta_steps=16):
        self.patch_size = 2 * int(finger_size / pixels_to_m) + 2
        self.finger_max_spread = int(finger_max_spread / pixels_to_m)
        self.step = step
        self.theta_steps = theta_steps

    def __call__(self, heightmap, mask, plot=False):
        heightmap_ = np.ones(heightmap.shape, dtype=np.bool)
        heightmap_[heightmap > 0] = False
        mask_ = np.ones(mask.shape, dtype=np.bool)
        mask_[mask > 0] = False
        # plt.imshow(heightmap_, cmap='gray', vmin=np.min(heightmap_), vmax=np.max(heightmap_))
        # plt.show()
        # plt.imshow(mask_, cmap='gray', vmin=np.min(mask_), vmax=np.max(mask_))
        # plt.show()

        theta = 0
        theta_ = None
        while theta < np.pi:
            center_1 = self._get_obstacle_free_position(mask_, theta)
            center_2 = self._get_obstacle_free_position(mask_, np.pi + theta)
            theta += np.pi / self.theta_steps
            if np.linalg.norm(center_1 - center_2) < self.finger_max_spread:
                if self.patch(heightmap_, center_1).all() and self.patch(heightmap_, center_2).all():
                    theta_ = theta
                    break

        if plot:
            fig, ax = plt.subplots(1)
            ax.imshow(heightmap, cmap='gray', vmin=np.min(heightmap), vmax=np.max(heightmap))
            if theta_ is not None:
                patch_x = int(center_1[0] - self.patch_size / 2.)
                patch_y = int(center_1[1] - self.patch_size / 2.)
                rect = patches.Rectangle((patch_y, patch_x), self.patch_size, self.patch_size, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                patch_x = int(center_2[0] - self.patch_size / 2.)
                patch_y = int(center_2[1] - self.patch_size / 2.)
                rect = patches.Rectangle((patch_y, patch_x), self.patch_size, self.patch_size, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show()

        return theta_

    def _get_obstacle_free_position(self, image, theta):
        center = np.array((int(image.shape[0] / 2), int(image.shape[1] / 2)))
        r = 0
        patch_center_result = None
        while True:
            patch_center = (center + r * np.array([np.sin(-theta), cos(-theta)])).astype(np.int32)
            try:
                if self.patch(image, patch_center).all():
                    patch_center_result = patch_center
                    break
            except AssertionError:
                break
            r += self.step
        return patch_center_result

    def patch(self, map, center, plot=False):
        """Check if the patch on the image is free from objects"""
        patch_x = int(center[0] - self.patch_size / 2.)
        patch_y = int(center[1] - self.patch_size / 2.)
        assert patch_x >= 0 and patch_y >= 0, 'Patch out of limits.'
        assert patch_x + self.patch_size < map.shape[1] and patch_y + self.patch_size < map.shape[0], 'Patch out of limits.'
        patch_image = map[patch_x:patch_x + self.patch_size, patch_y:patch_y + self.patch_size]
        if plot:
            fig, ax = plt.subplots(2)
            ax[0].imshow(patch_image, cmap='gray', vmin=np.min(patch_image), vmax=np.max(patch_image))
            ax[1].imshow(map, cmap='gray', vmin=np.min(map), vmax=np.max(map))
            rect = patches.Rectangle((patch_y, patch_x), self.patch_size, self.patch_size, linewidth=1, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect)
            plt.show()
        return patch_image

def get_table_point_cloud(pose, bbox, workspace, density=128, bbox_aug=0.008, plot=False):
    def in_hull(p, hull):
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p) < 0

    def get_corners(pose, bbox):
        bbox_corners_object = np.array([[bbox[0], bbox[1], bbox[2]],
                                        [bbox[0], -bbox[1], bbox[2]],
                                        [bbox[0], bbox[1], -bbox[2]],
                                        [bbox[0], -bbox[1], -bbox[2]],
                                        [-bbox[0], bbox[1], bbox[2]],
                                        [-bbox[0], -bbox[1], bbox[2]],
                                        [-bbox[0], bbox[1], -bbox[2]],
                                        [-bbox[0], -bbox[1], -bbox[2]]])
        pos = pose[0:3]
        quat = Quaternion(pose[3], pose[4], pose[5], pose[6])
        return transform_points(bbox_corners_object, pos, quat)

    x = np.linspace(-workspace[0], workspace[0], density)
    y = np.linspace(-workspace[1], workspace[1], density)
    x, y = np.meshgrid(x, y)
    x_y = np.column_stack((x.ravel(), y.ravel()))

    inhulls = np.ones((x_y.shape[0], pose.shape[0]), dtype=np.bool)
    for object in range(pose.shape[0]):
        corners = get_corners(pose[object], bbox[object] + bbox_aug)
        hull = ConvexHull(corners[:, 0:2])
        base_corners = corners[hull.vertices, 0:2]
        inhulls[:, object] = in_hull(x_y, base_corners)

    return x_y[inhulls.all(axis=1), :]

def preprocess_real_state(obs_dict, max_init_distance=0.1, angle=0, primitive=0):
    what_i_need = ['object_poses', 'object_bounding_box', 'object_above_table', 'surface_size', 'surface_edges',
                   'max_n_objects', 'init_distance_from_target', 'walls', 'n_fixed_objects', 'fixed_object_poses',
                   'fixed_object_bounding_box']
    state = {}
    for key in what_i_need:
        state[key] = copy.deepcopy(obs_dict[key])

    # Keep closest objects
    poses = state['object_poses'][state['object_above_table']]
    if poses.shape[0] == 0:
        return state
    threshold = max(max_init_distance, obs_dict['push_distance_range'][1]) + np.max(state['object_bounding_box'][0]) + \
                obs_dict['singulation_distance'][0]
    objects_close_target = np.linalg.norm(poses[:, 0:3] - poses[0, 0:3], axis=1) < threshold
    state['object_poses'] = state['object_poses'][state['object_above_table']][objects_close_target]
    state['object_bounding_box'] = state['object_bounding_box'][state['object_above_table']][objects_close_target]
    state['object_above_table'] = state['object_above_table'][state['object_above_table']][objects_close_target]

    # Filter out objects for the primitive
    target_bounding_box_z = obs_dict['target_bounding_box'][2]
    finger_height = obs_dict['finger_height']
    if primitive == 0:
        threshold = 0  # assuming the env does not spawn flat objects
    elif primitive == 1:
        threshold = 2 * target_bounding_box_z + 1.1 * finger_height
    else:
        raise ValueError()
    n_objects = state['object_poses'].shape[0]
    objects_useful_for_primitive = np.zeros(n_objects, dtype=np.bool)
    objects_useful_for_primitive[0] = True
    for i in range(1, n_objects):
        pose = state['object_poses'][i]
        bbox = state['object_bounding_box'][i]
        height = get_object_height(pose, bbox)
        if height > threshold:
            objects_useful_for_primitive[i] = True

    state['object_poses'] = state['object_poses'][objects_useful_for_primitive]
    state['object_bounding_box'] = state['object_bounding_box'][objects_useful_for_primitive]
    state['object_above_table'] = state['object_above_table'][objects_useful_for_primitive]

    app = np.zeros((4, 5))
    app[:, 1] = 1
    surface_edges = np.append(state['surface_edges'], app, axis=1)
    # Rotate
    poses = state['object_poses']
    target_pose = poses[0].copy()
    poses = transform_poses(poses, target_pose)
    surface_edges = transform_poses(surface_edges, target_pose)
    rotz = np.zeros(7)
    rotz[3:7] = Quaternion.from_rotation_matrix(rot_z(-angle)).as_vector()
    poses = transform_poses(poses, rotz)
    surface_edges = transform_poses(surface_edges, rotz)
    poses = transform_poses(poses, target_pose, target_inv=True)
    surface_edges = transform_poses(surface_edges, target_pose, target_inv=True)
    target_pose[3:] = np.zeros(4)
    target_pose[3] = 1
    poses = transform_poses(poses, target_pose)
    poses[:, 2] += target_pose[2]
    surface_edges = transform_poses(surface_edges, target_pose)
    surface_edges[:, 2] += target_pose[2]
    state['surface_edges'] = surface_edges[:, :2].copy()

    if obs_dict['walls']:
        distances = get_distances_from_walls(obs_dict)
    else:
        distances = [obs_dict['surface_size'][0] - obs_dict['object_poses'][0, 0], \
                     obs_dict['surface_size'][0] + obs_dict['object_poses'][0, 0], \
                     obs_dict['surface_size'][1] - obs_dict['object_poses'][0, 1], \
                     obs_dict['surface_size'][1] + obs_dict['object_poses'][0, 1]]

    state['surface_distances'] = np.array([x / 0.5 for x in distances])

    state['object_poses'] = poses
    shape = state['object_poses'].shape
    if shape[0] < state['max_n_objects']:
        state['object_poses'] = np.append(state['object_poses'],
                                                 np.zeros((int(state['max_n_objects']) - shape[0], shape[1])), axis=0)

    shape = state['object_bounding_box'].shape
    if shape[0] < state['max_n_objects']:
        state['object_bounding_box'] = np.append(state['object_bounding_box'],
                                                 np.zeros((int(state['max_n_objects']) - shape[0], shape[1])), axis=0)

    shape = state['object_above_table'].shape
    if shape[0] < state['max_n_objects']:
        state['object_above_table'] = np.append(state['object_above_table'],
                                                np.zeros((int(state['max_n_objects']) - shape[0]),
                                                         dtype=np.bool), axis=0)
    return state

def plot_real_state(state):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from robamine.utils.viz import plot_boxes, plot_frames
    fig = plt.figure()
    ax = Axes3D(fig)
    plot_boxes(state['object_poses'][state['object_above_table']][:, 0:3],
               state['object_poses'][state['object_above_table']][:, 3:7],
               state['object_bounding_box'][state['object_above_table']], ax)
    plot_frames(state['object_poses'][state['object_above_table']][:, 0:3],
                state['object_poses'][state['object_above_table']][:, 3:7], 0.01, ax)
    ax.scatter(state['surface_edges'][:, 0], state['surface_edges'][:, 1])
    ax.axis('equal')
    plt.show()


def obs_dict2feature(primitive, obs_dict, angle=0, real_state=False):
    if real_state:
        return RealState(obs_dict=obs_dict, angle=0)
    else:
        if primitive == 0:
            return PushTargetFeature(obs_dict=obs_dict, angle=angle)
        if primitive == 1:
            return PushObstacleFeature(obs_dict=obs_dict, angle=angle)
        if primitive == 2:
            return GraspTargetFeature(obs_dict=obs_dict, angle=angle)

    raise ValueError('Primitive should be 0, 1 or 2.')


def get_rotated_transition(transition, angle=0):
    state = obs_dict2feature(transition.action[0], transition.state, angle).array()
    if transition.next_state is not None:
        next_state = obs_dict2feature(transition.action[0], transition.next_state, angle).array().copy()
    else:
        next_state = None

    action = transition.action.copy()
    action[1] += min_max_scale(angle, range=[-180, 180], target_range=[-1, 1])
    if action[1] > 1:
        action[1] = -1 + abs(1 - action[1])

    return Transition(state=state.copy(),
                      action=action.copy(),
                      reward=transition.reward,
                      next_state=next_state,
                      terminal=transition.terminal)


def get_distance_of_two_bbox(pose_1, bbox_1, pose_2, bbox_2, density=0.005, plot=False):
    '''Calculates the distance between two oriented bounding boxes using point clouds.'''
    point_cloud_1 = discretize_3d_box(bbox_1[0], bbox_1[1], bbox_1[2], density)
    point_cloud_2 = discretize_3d_box(bbox_2[0], bbox_2[1], bbox_2[2], density)

    point_cloud_1 = transform_list_of_points(point_cloud_1, pose_1[0:3],
                                              Quaternion(pose_1[3], pose_1[4], pose_1[5], pose_1[6]))
    point_cloud_2 = transform_list_of_points(point_cloud_2, pose_2[0:3],
                                              Quaternion(pose_2[3], pose_2[4], pose_2[5], pose_2[6]))

    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(point_cloud_1[:, 0], point_cloud_1[:, 1], point_cloud_1[:, 2], marker='o')
        ax.scatter(point_cloud_2[:, 0], point_cloud_2[:, 1], point_cloud_2[:, 2], marker='o')
        ax.axis('equal')
        plt.show()

    return np.min(cdist(point_cloud_1, point_cloud_2))


def transform_list_of_points(points, pos, quat, inv=False):
    '''Points are w.r.t. {A}. pos and quat is the frame {A} w.r.t {B}. Returns the list of points experssed w.r.t.
    {B}.'''
    assert points.shape[1] == 3
    matrix = np.eye(4)
    matrix[0:3, 3] = pos
    matrix[0:3, 0:3] = quat.rotation_matrix()
    if inv:
        matrix = np.linalg.inv(matrix)

    transformed_points = np.transpose(np.matmul(matrix, np.transpose(
        np.concatenate((points, np.ones((points.shape[0], 1))), axis=1))))[:, :3]
    return transformed_points


def discretize_2d_box(x, y, density):
    assert x > 0 and y > 0

    xx = np.linspace(-x, x, int(2 * x / density))
    yy = np.linspace(-y, y, int(2 * y / density))
    xx, yy = np.meshgrid(xx, yy)
    out = np.zeros((int(2 * x / density) * int(2 * y / density), 3))
    out[:, 0] = xx.flatten()
    out[:, 1] = yy.flatten()
    return out


def discretize_3d_box(x, y, z, density):
    combos = [[x, y, z, ''],
              [x, y, -z, ''],
              [z, y, -x, 'y',],
              [z, y, x, 'y'],
              [x, z, y, 'x'],
              [x, z, -y, 'x']]
    faces = []
    for combo in combos:
        face = discretize_2d_box(combo[0], combo[1], density)
        face[:, 2] = combo[2]
        if combo[3] == 'y':
            rot = rot_y(pi / 2)
            face = np.transpose(np.matmul(rot, np.transpose(face)))
        elif combo[3] == 'x':
            rot = rot_x(pi / 2)
            face = np.transpose(np.matmul(rot, np.transpose(face)))
        faces.append(face)
    result = np.concatenate(faces, axis=0)
    return result


def plot_point_cloud_of_scene(obs_dict, density=0.005):
    n_objects = int(obs_dict['n_objects'])
    target_pose = obs_dict['object_poses'][0]
    target_bbox = obs_dict['object_bounding_box'][0]
    point_cloud_1 = discretize_3d_box(target_bbox[0], target_bbox[1], target_bbox[2], density)
    point_cloud_1 = transform_list_of_points(point_cloud_1, target_pose[0:3],
                                             Quaternion(target_pose[3], target_pose[4], target_pose[5], target_pose[6]))
    point_clouds = []
    point_clouds.append(point_cloud_1)
    for i in range(1, n_objects):
        obstacle_pose = obs_dict['object_poses'][i]
        obstacle_bbox = obs_dict['object_bounding_box'][i]
        point_cloud_2 = discretize_3d_box(obstacle_bbox[0], obstacle_bbox[1], obstacle_bbox[2], density)

        point_cloud_2 = transform_list_of_points(point_cloud_2, obstacle_pose[0:3],
                                                 Quaternion(obstacle_pose[3], obstacle_pose[4], obstacle_pose[5],
                                                            obstacle_pose[6]))
        point_clouds.append(point_cloud_2)
    point_clouds = np.concatenate(point_clouds, axis=0)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(point_clouds[:, 0], point_clouds[:, 1], point_clouds[:, 2], marker='o')
    # ax.axis('equal')
    plt.show()


def is_object_above_object(pose, bbox, pose_2, bbox_2, density=0.005, plot=False):
    """
    Test if object with pose `pose_2` and bounding box `bbox_2`
    is above the object with pose `pose` and bounding box `bbox`
    """
    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p) >= 0

    def get_corners(pose, bbox):
        """
        Return the 8 points of the bounding box
        """
        bbox_corners_object = np.array([[bbox[0], bbox[1], bbox[2]],
                                        [bbox[0], -bbox[1], bbox[2]],
                                        [bbox[0], bbox[1], -bbox[2]],
                                        [bbox[0], -bbox[1], -bbox[2]],
                                        [-bbox[0], bbox[1], bbox[2]],
                                        [-bbox[0], -bbox[1], bbox[2]],
                                        [-bbox[0], bbox[1], -bbox[2]],
                                        [-bbox[0], -bbox[1], -bbox[2]]])
        pos = pose[0:3]
        quat = Quaternion(pose[3], pose[4], pose[5], pose[6])
        return transform_list_of_points(bbox_corners_object, pos, quat)

    corners_1 = get_corners(pose, bbox)
    hull = ConvexHull(corners_1[:, 0:2])
    base_corners = corners_1[hull.vertices, 0:2]

    if plot:
        plt.scatter(base_corners[:, 0], base_corners[:, 1], color=[1, 0, 0])
    point_cloud = discretize_3d_box(bbox_2[0], bbox_2[1], bbox_2[2], density=density)
    pos = pose_2[0:3]
    quat = Quaternion(pose_2[3], pose_2[4], pose_2[5], pose_2[6])
    point_cloud = transform_list_of_points(point_cloud, pos, quat)
    if plot:
        plt.scatter(point_cloud[:, 0], point_cloud[:, 1], color=[0, 1, 0])
        plt.show()
    return in_hull(point_cloud[:, 0:2], base_corners).any()

# def get_table_point_cloud(pose, bbox, workspace, density=128, bbox_aug=0.008, plot=False):
#     def in_hull(p, hull):
#         if not isinstance(hull, Delaunay):
#             hull = Delaunay(hull)
#
#         return hull.find_simplex(p) < 0
#
#     def get_corners(pose, bbox):
#         bbox_corners_object = np.array([[bbox[0], bbox[1], bbox[2]],
#                                         [bbox[0], -bbox[1], bbox[2]],
#                                         [bbox[0], bbox[1], -bbox[2]],
#                                         [bbox[0], -bbox[1], -bbox[2]],
#                                         [-bbox[0], bbox[1], bbox[2]],
#                                         [-bbox[0], -bbox[1], bbox[2]],
#                                         [-bbox[0], bbox[1], -bbox[2]],
#                                         [-bbox[0], -bbox[1], -bbox[2]]])
#         pos = pose[0:3]
#         quat = Quaternion(pose[3], pose[4], pose[5], pose[6])
#         return transform_list_of_points(bbox_corners_object, pos, quat)
#
#     x = np.linspace(-workspace[0], workspace[0], density)
#     y = np.linspace(-workspace[1], workspace[1], density)
#     x, y = np.meshgrid(x, y)
#     x_y = np.column_stack((x.ravel(), y.ravel()))
#
#     inhulls = np.ones((x_y.shape[0], pose.shape[0]), dtype=np.bool)
#     for object in range(pose.shape[0]):
#         corners = get_corners(pose[object], bbox[object] + bbox_aug)
#         hull = ConvexHull(corners[:, 0:2])
#         base_corners = corners[hull.vertices, 0:2]
#         inhulls[:, object] = in_hull(x_y, base_corners)
#
#     return x_y[inhulls.all(axis=1), :]


def predict_collision(obs, x, y, theta, walls=False):
    if walls and (abs(x) > obs['surface_size'][0] - obs['finger_length'][0]  or abs(y) > obs['surface_size'][1] - obs['finger_length'][0]):
        return True

    n_objects = int(len(obs['object_poses'][obs['object_above_table']]))
    sphere_pose = np.zeros(7)
    sphere_pose[0] = x
    sphere_pose[1] = y
    sphere_pose[2] = 0.5
    sphere_pose[3:] = Quaternion.from_rotation_matrix(rot_z(theta)).as_vector()
    sphere_bbox = obs['finger_length'][0] * np.ones(3)
    for i in range(0, n_objects):
        object_pose = obs['object_poses'][obs['object_above_table']][i]
        object_bbox = obs['object_bounding_box'][obs['object_above_table']][i]
        if is_object_above_object(object_pose, object_bbox, sphere_pose, sphere_bbox, density=0.0025):
            return True
    return False


class ObstacleAvoidanceLoss(nn.Module):
    def __init__(self, distance_range, min_dist_range=[0.002, 0.1], device='cpu'):
        super(ObstacleAvoidanceLoss, self).__init__()
        self.distance_range = distance_range
        self.min_dist_range = min_dist_range
        self.device = device

    def forward(self, point_clouds, actions):
        # Transform the action to cartesian
        theta = min_max_scale(actions[:, 0], range=[-1, 1], target_range=[-pi, pi], lib='torch', device=self.device)
        distance = min_max_scale(actions[:, 1], range=[-1, 1], target_range=self.distance_range, lib='torch',
                                 device=self.device)
        # TODO: Assumes 2 actions!
        x_y = torch.zeros(actions.shape).to(self.device)
        x_y[:, 0] = distance * torch.cos(theta)
        x_y[:, 1] = distance * torch.sin(theta)
        x_y = x_y.reshape(x_y.shape[0], 1, x_y.shape[1]).repeat((1, point_clouds.shape[1], 1))
        diff = x_y - point_clouds
        min_dist = torch.min(torch.norm(diff, p=2, dim=2), dim=1)[0]
        threshold = torch.nn.Threshold(threshold=- self.min_dist_range[1], value= - self.min_dist_range[1])
        min_dist = - threshold(- min_dist)
        # hard_shrink = torch.nn.Hardshrink(lambd=self.min_dist_range[0])
        # min_dist = hard_shrink(min_dist)
        # print('mindist', min_dist)
        obstacle_avoidance_signal = - min_max_scale(min_dist, range=self.min_dist_range, target_range=[0.0, 5],
                                                    lib='torch', device=self.device)
        # print('signal: ', obstacle_avoidance_signal)
        close_center_signal = 0.5 - min_max_scale(distance, range=self.distance_range, target_range=[0, .5], lib='torch',
                                                  device=self.device)
        # final_signal = close_center_signal
        final_signal = obstacle_avoidance_signal
        return - final_signal.mean()

    def plot(self, point_cloud, density=64):
        from mpl_toolkits.mplot3d import Axes3D
        x_min = np.min(point_cloud[:, 0])
        x_max = np.max(point_cloud[:, 0])
        y_min = np.min(point_cloud[:, 1])
        y_max = np.max(point_cloud[:, 1])
        print(x_min, x_max, y_min, y_max)
        point_cloud = torch.FloatTensor(point_cloud).to(self.device)

        x = np.linspace(x_min, x_max, density)
        y = np.linspace(y_min, y_max, density)
        x_, y_ = np.meshgrid(x, y)
        z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                theta = min_max_scale(np.arctan2(y_[i, j], x_[i, j]), range=[-np.pi, np.pi], target_range=[-1, 1])
                distance = min_max_scale(np.sqrt(y_[i, j] ** 2 + x_[i, j] ** 2), range=[0, max(x_max, y_max)],
                                         target_range=[-1, 1])
                z[i, j] = self.forward(point_cloud.reshape((1, point_cloud.shape[0], -1)),
                                       torch.FloatTensor([theta, distance]).reshape(1, -1))

        # Uncomment to print min value
        # ij = np.argwhere(z == np.min(z))
        # print(':', x_[ij[0,0], ij[0,1]], y_[ij[0, 0], ij[0, 1]])
        fig = plt.figure()
        axs = Axes3D(fig)
        mycmap = plt.get_cmap('winter')
        surf1 = axs.plot_surface(x_, y_, z, cmap=mycmap)
        fig.colorbar(surf1, ax=axs, shrink=0.5, aspect=5)
