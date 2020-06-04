import numpy as np
from math import cos, sin, pi, acos, atan2
from robamine.utils.math import min_max_scale, cartesian2spherical
from robamine.utils.orientation import rot_z, Quaternion, transform_poses, transform_points
import matplotlib.pyplot as plt
from robamine.clutter.mdp import Feature, PushTargetWithObstacleAvoidance, PushAction2D, PushTarget2D
from robamine.utils.math import LineSegment2D
import copy
from scipy.spatial import ConvexHull, Delaunay

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
        array = np.append(array, self.surface_edges.flatten())
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
        return 10 * 4 * 3 + 8 + 1 # TODO: hardcoded max n objects

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

        super(PushTargetRealWithObstacleAvoidance, self).__init__(theta=theta, push_distance=push_distance, distance=distance,
                                         push_distance_range=push_distance_range,
                                         init_distance_range=init_distance_range,
                                         convex_hull=convex_hull, object_height=object_height, finger_size=finger_size)

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
    def __init__(self, obs_dict, angle, push_distance, push_distance_range, init_distance_range, target_height, finger_size, eps=0.05):
        angle_ = min_max_scale(angle, range=[-1, 1], target_range=[-np.pi, np.pi])
        push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=push_distance_range)
        eps_ = min_max_scale(eps, range=[-1, 1], target_range=[-np.pi, np.pi])

        max_init_distance = init_distance_range[1]
        x_init = max_init_distance * np.cos(angle_)
        y_init = max_init_distance * np.sin(angle_)
        preprocessed = preprocess_real_state(obs_dict, max_init_distance, 0)

        point_cloud = self.get_table_point_cloud(preprocessed['object_poses'][preprocessed['object_above_table']],
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

        push_avoid = PushTargetRealWithObstacleAvoidance(obs_dict, angle, push_distance=-1, distance=-1, push_distance_range=[0, 0.1], init_distance_range=[0, 0.1])
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

    def get_table_point_cloud(self, pose, bbox, workspace, density=128, bbox_aug=0.008, plot=False):
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

def preprocess_real_state(obs_dict, max_init_distance=0.1, angle=0):
    what_i_need = ['object_poses', 'object_bounding_box', 'object_above_table', 'surface_size', 'surface_edges',
                   'max_n_objects', 'init_distance_from_target']
    state = {}
    for key in what_i_need:
        state[key] = copy.deepcopy(obs_dict[key])

    # Keep closest objects
    poses = state['object_poses'][state['object_above_table']]
    if poses.shape[0] == 0:
        return state

    app = np.zeros((4, 5))
    app[:, 1] = 1
    surface_edges = np.append(state['surface_edges'], app, axis=1)
    threshold = max(max_init_distance, obs_dict['push_distance_range'][1]) + np.max(state['object_bounding_box'][0]) + obs_dict['singulation_distance'][0]
    objects_close_target = np.linalg.norm(poses[:, 0:3] - poses[0, 0:3], axis=1) < threshold
    state['object_poses'] = state['object_poses'][state['object_above_table']][objects_close_target]
    state['object_bounding_box'] = state['object_bounding_box'][state['object_above_table']][objects_close_target]
    state['object_above_table'] = state['object_above_table'][state['object_above_table']][objects_close_target]
    # Rotate
    poses = state['object_poses'][state['object_above_table']]
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
    surface_edges = transform_poses(surface_edges, target_pose)
    state['surface_edges'] = surface_edges[:, :2].copy()
    state['object_poses'][state['object_above_table']] = poses
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
