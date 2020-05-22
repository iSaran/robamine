import numpy as np
from math import cos, sin, pi, acos, atan2
from robamine.utils.math import min_max_scale, cartesian2spherical
from robamine.utils.orientation import rot_z, Quaternion
import matplotlib.pyplot as plt
from robamine.clutter.mdp import FeatureBase, PushActionBase
from robamine.utils.math import LineSegment2D

class RealState(FeatureBase):
    def __init__(self, obs_dict, angle=0, sort=True, normalize=True, spherical=True, range_norm=[-1, 1], name=None):
        '''Angle in rad.'''
        self.poses = obs_dict['object_poses'][obs_dict['object_above_table']]
        self.bounding_box = obs_dict['object_bounding_box'][obs_dict['object_above_table']]
        self.n_objects = self.poses.shape[0]
        self.max_n_objects = obs_dict['max_n_objects']
        self.range_norm = range_norm

        # Calculate principal corners for feature
        self.principal_corners = np.zeros((self.n_objects, 4, 3))
        self.calculate_principal_corners()
        self.principal_corners_plot = self.principal_corners.copy()
        self.rotate(angle)
        self.surface_size = obs_dict['surface_size']
        self.coordinates = 'cartesian'
        if spherical:
            self.coordinates = 'spherical'
            init_shape = self.principal_corners.shape
            self.principal_corners = cartesian2spherical(self.principal_corners.reshape(-1, 3)).reshape(init_shape)
        if sort:
            self.sort()

        # Append for table limits
        rot_2d = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        self.target_pos = np.matmul(rot_2d, self.poses[0, 0:2])
        self.surface_edge = np.matmul(rot_2d, self.surface_size)

        if normalize:
            self.normalize()

    def calculate_principal_corners(self):
        self.principal_corners[0] = self._get_principal_corner_target()
        for i in range(1, self.n_objects):
            self.principal_corners[i, :, :] = self.get_principal_corner_obstacle(i)

    def get_principal_corner_obstacle(self, i):
        pos_target = self.poses[0, 0:3]
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
        principal_corners_target = self._transform_list_of_points(principal_corners_world, np.append(pos_target[0:2], 0), Quaternion(), inv=True)

        # bbox_corners_target_rotated = np.matmul(rot_z(angle))
        return principal_corners_target

    def _get_principal_corner_target(self):
        pos = self.poses[0, 0:3]
        quat = Quaternion(self.poses[0, 3], self.poses[0, 4], self.poses[0, 5], self.poses[0, 6])
        bbox = self.bounding_box[0]
        bbox[2] = -bbox[2]

        principal_corners_object = self._get_the_other_3_principal_corners(bbox)
        principal_corners_world = self._transform_list_of_points(principal_corners_object, pos=pos, quat=quat)
        # Mujoco due to soft contacts has bottom under 0. Enforce to zero
        principal_corners_world[:, 2][principal_corners_world[:, 2] < 0] = 0
        principal_corners_target = self._transform_list_of_points(principal_corners_world, pos=np.append(self.poses[0, 0:2], 0),
                                                                  quat=Quaternion(), inv=True)
        return principal_corners_target

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
        elif self.coordinates == 'spherical':
            range_ = [[0, np.linalg.norm(self.surface_size) + eps],
                      [-np.pi, np.pi],
                      [-np.pi, np.pi]]

        for i in range(3):
            self.principal_corners[:, :, i] = min_max_scale(self.principal_corners[:, :, i],
                                                            range=range_[i],
                                                            target_range=self.range_norm)

        max_surface_edge = np.linalg.norm(self.surface_size) + eps
        self.target_pos = min_max_scale(self.target_pos, range=[-max_surface_edge, max_surface_edge],
                                        target_range=self.range_norm)
        self.surface_edge = min_max_scale(self.surface_edge, range=[-max_surface_edge, max_surface_edge],
                                          target_range=self.range_norm)

    def array(self):
        array = np.concatenate((self.principal_corners, self.range_norm[0] * np.ones(
            (int(self.max_n_objects - self.principal_corners.shape[0]), 4, 3)))).flatten()
        array = np.append(array, self.target_pos)
        array = np.append(array, self.surface_edge)


        # assert (array <= 1).all()
        # assert (array >= -1).all()

        # assert array.shape[0] == self.dim()
        return array

    def rotate(self, angle):
        for i in range(self.n_objects):
            self.principal_corners[i] = self._transform_list_of_points(self.principal_corners[i], pos=np.zeros(3), quat=Quaternion.from_rotation_matrix(rot_z(angle)))

    def plot(self, centroids=False, action=None):
        from mpl_toolkits.mplot3d import Axes3D
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
        plt.show()

    @staticmethod
    def dim():
        return 10 * 4 * 3 + 4 # TODO: hardcoded max n objects


class PushTarget(PushActionBase):
    def __init__(self, obs_dict, theta, push_distance, distance, push_distance_range, init_distance_range):
        # Get the vertices of the bounding box from real state
        self.theta = min_max_scale(theta, range=[-1, 1], target_range=[-pi, pi])
        self.push_distance = push_distance
        self.distance = distance

        self.push_distance_range = push_distance_range
        self.init_distance_range = init_distance_range
        self.target_principal_corners = RealState(obs_dict, sort=False, normalize=False, spherical=False).principal_corners[0]

        self.convex_hull = [LineSegment2D(self.target_principal_corners[0][0:2], self.target_principal_corners[1][0:2]),
                            LineSegment2D(self.target_principal_corners[0][0:2], self.target_principal_corners[2][0:2]),
                            LineSegment2D(-self.target_principal_corners[0][0:2], self.target_principal_corners[1][0:2]),
                            LineSegment2D(-self.target_principal_corners[0][0:2], self.target_principal_corners[2][0:2])]

        finger_size = obs_dict['finger_height']
        self.distance_line_segment = self._get_distance_line_segment(finger_size)
        self.push_line_segment = self._get_push_line_segment()

        object_height = self.target_principal_corners[3][2] / 2
        if object_height - finger_size > 0:
            offset = object_height - finger_size
        else:
            offset = 0
        self.z = finger_size + offset + 0.001

    def _get_push_line_segment(self):
        direction = np.array([cos(self.theta + pi), sin(self.theta + pi)])
        line_segment = LineSegment2D(np.zeros(2), 10 * direction)
        min_point = np.zeros(2)
        max_point = self.push_distance_range[1] * direction
        return LineSegment2D(min_point, max_point)

    def _get_distance_line_segment(self, finger_size):
        # Calculate the intersection point between the direction of the push theta and the convex hull (four line segments)
        direction = np.array([cos(self.theta), sin(self.theta)])
        line_segment = LineSegment2D(np.zeros(2), 10 * direction)
        min_point = line_segment.get_first_intersection_point(self.convex_hull)
        min_point += (finger_size + 0.008) * direction
        max_point = (np.linalg.norm(min_point) + self.init_distance_range[1]) * direction
        return LineSegment2D(min_point, max_point)

    def get_init_pos(self):
        lambd = min_max_scale(self.distance, range=[-1, 1], target_range=[0, 1])
        init = self.distance_line_segment.get_point(lambd)
        return np.append(init, self.z)

    def get_final_pos(self):
        lambd = min_max_scale(self.push_distance, range=[-1, 1], target_range=[0, 1])
        init = self.push_line_segment.get_point(lambd)
        return np.append(init, self.z)

    def get_duration(self, distance_per_sec = 0.1):
        return np.linalg.norm(self.get_init_pos() - self.get_final_pos()) / distance_per_sec

