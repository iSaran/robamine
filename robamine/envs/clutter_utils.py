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

from robamine.utils.math import LineSegment2D, triangle_area, min_max_scale
from robamine.utils.orientation import rot_x, rot_z, rot2angleaxis
from robamine.utils.cv_tools import Feature
from robamine.utils.info import get_now_timestamp

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from robamine.algo.core import InvalidEnvError


from robamine.algo.util import Transition


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
    action_dim_all = [3, 1, 1]

    if primitive >= 0:
        action_dim = [action_dim_all[primitive]]
    else:
        action_dim = action_dim_all

    return action_dim


def get_observation_dim(primitive):
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


def obs_dict2feature(primitive, obs_dict, angle=0):
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
