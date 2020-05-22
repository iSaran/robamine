import numpy as np
from scipy.spatial.distance import cdist
from robamine.utils.orientation import Quaternion, rot_x, rot_y, rot_z


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