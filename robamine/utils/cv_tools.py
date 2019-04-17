#!/usr/bin/env python3
import numpy as np
import cv2
import open3d

'''
Computer Vision Utils
============
'''


def depth_to_point_cloud(depth, camera_intrinsics):
    """
    Converts a depth map to a point cloud(
    :param depth: depth image
    :param camera_intrinsics: focal length and center point
    :return: nx3 numpy array
    """
    fx = camera_intrinsics[0]
    fy = camera_intrinsics[1]
    cx = camera_intrinsics[2]
    cy = camera_intrinsics[3]

    point_cloud = []

    h, w = depth.shape
    for x in range(0, w):
        for y in range(0, h):
            if depth[y][x] != 0:
                Z = -depth[y][x]  # z is negative because Z axis is inward
                X = (x - cx) * Z / fx
                Y = (y - cy) * Z / fy
                point_cloud.append([X, Y, Z])

    return np.asarray(point_cloud)


def transform_point_cloud(point_cloud, affine_transformation):
    """
    Apply an affine transformation to the point cloud
    :param point_cloud: input point cloud
    :param affine_transformation: 4x4 matrix that describes the affine transformation [R|t]
    :return:
    """
    # Convert cartesian to homogeneous coordinates
    ones = np.ones((point_cloud.shape[0], 1), dtype=np.float32)
    point_cloud = np.concatenate((point_cloud, ones), axis=1)

    # Transform cloud
    for i in range(point_cloud.shape[0]):
        point_cloud[i] = np.matmul(affine_transformation, point_cloud[i])
    # point_cloud = np.matmul(affine_transformation, np.transpose(point_cloud))

    # Convert homogeneous to cartesian
    w = point_cloud[:, 3]
    point_cloud /= w[:, np.newaxis]

    return point_cloud[:, 0:3]


def gl2cv(depth, z_near, z_far):
    """

    :param depth:
    :param z_near:
    :param z_far:
    :return:
    """
    h, w = depth.shape
    linear_depth = np.zeros((h, w), dtype=np.float32)

    for x in range(0, w):
        for y in range(0, h):
            if depth[y][x] != 1:
                linear_depth[y][x] = 2 * z_far * z_near / (z_far + z_near - (z_far - z_near) * (2 * depth[y][x] - 1))

    return np.flip(linear_depth, axis=0)


def rgb2bgr(rgb):
    """
    Convert a rgb image to bgr
    (Vertical flipping of the image)
    :param rgb: the image in bgr format
    """
    h, w, c = rgb.shape

    bgr = np.zeros((h, w, c), dtype=np.uint8)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    bgr[:, :, 0] = b
    bgr[:, :, 1] = g
    bgr[:, :, 2] = r
    return np.flip(bgr, axis=0)


def generate_height_map(point_cloud, plot=False):
    """

    :param point_cloud:
    :param plot:
    :return:
    """
    width = 100
    height = 100
    grid_step = 0.005

    height_grid = np.zeros((height, width), dtype=np.float32)

    for i in range(point_cloud.shape[0]):
        x = point_cloud[i][0]
        y = point_cloud[i][1]
        z = point_cloud[i][2]

        idx_x = int(np.floor(x / grid_step)) + int(width / 2)
        idx_y = int(np.floor(y / grid_step)) + int(height / 2)

        if 0 < idx_x < width - 1 and 0 < idx_y < height - 1:
            if height_grid[idx_y][idx_x] < z:
                height_grid[idx_y][idx_x] = z

    if plot:
        cv_height = np.zeros((height, width), dtype=np.float32)
        min_height = np.min(height_grid)
        max_height = np.max(height_grid)
        for i in range(0, width):
            for j in range(0, height):
                cv_height[i][j] = 255 * ((height_grid[i][j] - min_height) / (max_height - min_height))

        cv_height = np.flip(cv_height, axis=0)
        cv_height = np.flip(cv_height, axis=1)
        cv2.imshow("height_grid", cv_height)
        cv2.waitKey()

    return height_grid


def extract_features(height_map, bbox):
    """
    Extract features from height map
    :param height_map: height map aligned with the target
    :param bbox: target's dimensions
    :return: N-d vector
    """
    boxes = []

    h, w = height_map.shape

    side = 1/2 * 50 * bbox
    cx = int(w/2)
    cy = int(h/2)

    # Targets features
    cx1 = cx - int(side[0])
    cx2 = cx + int(side[0])
    cy1 = cy - int(side[1])
    cy2 = cy + int(side[1])

    cv_height = np.zeros((h, w), dtype=np.float32)
    min_height = np.min(height_map)
    max_height = np.max(height_map)
    for i in range(0, w):
        for j in range(0, h):
            cv_height[i][j] = 255 * ((height_map[i][j] - min_height) / (max_height - min_height))

    cv_height = np.flip(cv_height, axis=0)
    cv_height = np.flip(cv_height, axis=1)
    rgb = cv2.cvtColor(cv_height, cv2.COLOR_GRAY2RGB)

    boxes.append([(cx, cy), (cx1, cy), (cx1, cy1), (cx, cy1)])
    boxes.append([(cx, cy), (cx2, cy), (cx2, cy1), (cx, cy1)])
    boxes.append([(cx, cy), (cx2, cy), (cx2, cy2), (cx, cy2)])
    boxes.append([(cx, cy), (cx1, cy), (cx1, cy2), (cx, cy2)])
    # for i in range(len(boxes)):
    #     rgb = draw_box(boxes[i], rgb)
    #     cv2.imshow("height_grid", rgb)
    #     cv2.waitKey()

    # Features around target
    sum = 0
    # int(cx - 16), int(cy - side[1] - 32) # f_up
    # int(cx + side[0]), int(cy - 16) # f_right
    # up_left = int(cx - 16), int(cy + side[1]) # f_down
    up_left = (int(cx - side[0] - 32), int(cy - 16)) #f_left
    c = up_left

    # for x in range(8):
    #     for y in range(8):
    #         c = (up_left[0] + x * 4, up_left[1] + y * 4)
    #         boxes.append([c, (c[0]+4, c[1]), (c[0]+4, c[1]+4), (c[0], c[1]+4)])

    for box in boxes:
        x1 = box[0][0]
        x2 = box[1][0]
        y1 = box[0][1]
        y2 = box[2][1]
        print(x1, x2, y1, y2)
        sum = np.sum(height_map[y1:y2, x1:x2])
        print(height_map[y1:y2, x1:x2])
        sum /= 16.0
        print(sum)
        rgb = draw_box(box, rgb)
        cv2.imshow('rgb', rgb)
        cv2.waitKey()


def draw_box(box, rgb):
    cv2.line(rgb, box[0], box[1], (0, 255, 0), thickness=1)
    cv2.line(rgb, box[1], box[2], (0, 255, 0), thickness=1)
    cv2.line(rgb, box[2], box[3], (0, 255, 0), thickness=1)
    cv2.line(rgb, box[3], box[0], (0, 255, 0), thickness=1)
    return rgb



def plot_point_cloud(point_cloud):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(point_cloud)
    open3d.draw_geometries([pcd])