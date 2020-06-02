"""
ClutterCont
=======

Clutter Env for continuous control
"""

import numpy as np
from math import cos, sin, pi, acos, atan2

from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from robamine.utils.math import LineSegment2D, triangle_area, min_max_scale, cartesian2spherical
from robamine.utils.orientation import rot_x, rot_z, rot2angleaxis, Quaternion, rot_y, transform_poses
from robamine.utils.cv_tools import Feature
from robamine.utils.info import get_now_timestamp
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from robamine.algo.core import InvalidEnvError

from scipy.spatial.distance import cdist

from robamine.algo.util import Transition
from robamine.clutter.real_mdp import RealState
import torch
import torch.nn as nn

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
    action_dim_all = [1, 1, 1]

    if primitive >= 0:
        action_dim = [action_dim_all[primitive]]
    else:
        action_dim = action_dim_all

    return action_dim


def get_observation_dim(primitive, real_state=False):
    if real_state:
        obs_dim_all = [RealState.dim(), RealState.dim(), RealState.dim()]
    else:
        obs_dim_all = [PushTargetFeature.dim(), PushObstacleFeature.dim(), GraspTargetFeature.dim()]

    if primitive >= 0:
        obs_dim = [obs_dim_all[primitive]]
    else:
        obs_dim = obs_dim_all

    return obs_dim


# Features for different primitives
# ---------------------------------




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
        feature = feature.crop(self.crop_area[0], self.crop_area[1])
        feature = feature.pooling(mode='AVG')
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


def get_distance_of_two_bbox(pose_1, bbox_1, pose_2, bbox_2, density=0.005):
    '''Calculates the distance between two oriented bounding boxes using point clouds.'''
    point_cloud_1 = discretize_3d_box(bbox_1[0], bbox_1[1], bbox_1[2], density)
    point_cloud_2 = discretize_3d_box(bbox_2[0], bbox_2[1], bbox_2[2], density)

    point_cloud_1 = transform_list_of_points(point_cloud_1, pose_1[0:3],
                                              Quaternion(pose_1[3], pose_1[4], pose_1[5], pose_1[6]))
    point_cloud_2 = transform_list_of_points(point_cloud_2, pose_2[0:3],
                                              Quaternion(pose_2[3], pose_2[4], pose_2[5], pose_2[6]))

    # Uncomment to plot point clouds
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker='o')
    # ax.axis('equal')
    # plt.show()

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
        return transform_list_of_points(bbox_corners_object, pos, quat)

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

def predict_collision(obs, x, y):
    n_objects = int(obs['n_objects'])
    sphere_pose = np.zeros(7)
    sphere_pose[0] = x
    sphere_pose[1] = y
    sphere_pose[2] = 0.5
    sphere_pose[3] = 1  # identity orientation
    sphere_bbox = obs['finger_height'][0] * np.ones(3)
    for i in range(0, n_objects):
        object_pose = obs['object_poses'][i]
        object_bbox = obs['object_bounding_box'][i]
        if is_object_above_object(object_pose, object_bbox, sphere_pose, sphere_bbox, density=0.0025):
            return True
    return False

def preprocess_real_state(obs_dict, max_init_distance, max_obs_bounding_box, angle=0):
    what_i_need = ['object_poses', 'object_bounding_box', 'object_above_table', 'surface_size', 'surface_angle',
                   'max_n_objects']

    state = {}
    for key in what_i_need:
        state[key] = copy.deepcopy(obs_dict[key])

    # Keep closest objects
    poses = state['object_poses'][state['object_above_table']]
    if poses.shape[0] == 0:
        return state

    objects_close_target = np.linalg.norm(poses[:, 0:3] - poses[0, 0:3], axis=1) < (
            max_init_distance + max_obs_bounding_box + 0.01)
    state['object_poses'] = state['object_poses'][state['object_above_table']][objects_close_target]
    state['object_bounding_box'] = state['object_bounding_box'][state['object_above_table']][objects_close_target]
    state['object_above_table'] = state['object_above_table'][state['object_above_table']][objects_close_target]
    # Rotate
    poses = state['object_poses'][state['object_above_table']]
    target_pose = poses[0].copy()
    poses = transform_poses(poses, target_pose)
    rotz = np.zeros(7)
    rotz[3:7] = Quaternion.from_rotation_matrix(rot_z(-angle)).as_vector()
    poses = transform_poses(poses, rotz)
    poses = transform_poses(poses, target_pose, target_inv=True)
    target_pose[3:] = np.zeros(4)
    target_pose[3] = 1
    poses = transform_poses(poses, target_pose)
    state['object_poses'][state['object_above_table']] = poses
    state['surface_angle'] = angle
    return state

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
