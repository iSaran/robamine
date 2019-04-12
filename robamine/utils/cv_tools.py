#!/usr/bin/env python3
import numpy as np
import cv2

'''
Computer Vision Utils
============
'''


def depth_to_point_cloud(depth, camera_intrinsics):
    """
    Converts a depth map to a point cloud
    :param depth: depth image
    :param camera_intrinsics: focal length and center point
    :return: nx3 numpy array
    """
    fx = camera_intrinsics[0]
    fy = camera_intrinsics[1]
    cx = camera_intrinsics[2]
    cy = camera_intrinsics[3]

    point_cloud = []

    w, h = depth.shape
    for i in range(0, w):
        for j in range(0, h):
            if depth[i][j] != 0:
                z = depth[i][j] / 1000.0
                x = (i - cx) * z / fx
                y = (j - cy) * z / fy
                point_cloud.append([x, y, z])

    return np.asarray(point_cloud)


# def plot_point_cloud(point_cloud):
#     pcd = PointCloud()
#     pcd.points = Vector3dVector(point_cloud)
#     draw_geometries([pcd]


def mj2opencv(mj_rgb, mj_depth):
    w, h, c = mj_rgb.shape

    cv_rgb = np.zeros((w, h, c), dtype=np.uint8)
    r = mj_rgb[:, :, 0]
    g = mj_rgb[:, :, 1]
    b = mj_rgb[:, :, 2]
    cv_rgb[:, :, 0] = b
    cv_rgb[:, :, 1] = g
    cv_rgb[:, :, 2] = r

    cv_depth = np.zeros((w, h), dtype=np.uint8)
    cv_depth = mj_depth

    return cv_rgb, cv_depth
