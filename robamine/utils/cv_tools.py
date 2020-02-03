#!/usr/bin/env python3
import numpy as np
import cv2
import open3d
import math

import torch
import torch.nn as nn
import torch.optim as optim
'''
Computer Vision Utils
============
'''

ae_params = {
    'layers': 4,
    'encoder': {
        'filters': [16, 32, 64, 128],
        'kernels': [3, 3, 3, 3],
    },
    'decoder': {
        'filters': [128, 64, 32, 16],
        'kernels': [4, 4, 4, 4],
    },
    'device': 'cpu'
}

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, params = ae_params):
        super(Encoder, self).__init__()
        h, w, c = input_dim

        self.filters = params['encoder']['filters']
        self.kernels = params['encoder']['kernels']
        self.no_of_layers = params['layers']

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(c, self.filters[0], self.kernels[0]))
        self.layers.append(nn.MaxPool2d(2, padding=1))

        for i in range(1, self.no_of_layers):
            self.layers.append(nn.Conv2d(self.filters[i-1], self.filters[i], self.kernels[i]))
            self.layers.append(nn.MaxPool2d(2, padding=1))

        self.layers.append(nn.Linear(8192, latent_dim))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = nn.functional.relu(self.layers[i](x))
            # x = nn.functional.relu(self.layers[i](x))

        x = x.view(-1, 8192)
        x = nn.functional.relu(self.layers[-1](x))
        return x

'''
Decoder without skip connections
'''
class Decoder(nn.Module):
    def __init__(self, latent_dim, out_dim, params = ae_params):
        super(Decoder, self).__init__()

        self.filters = params['decoder']['filters']
        self.kernels = params['decoder']['kernels']
        self.no_of_layers = params['layers']

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(latent_dim, 8192))

        for i in range(self.no_of_layers - 1):
            self.layers.append(nn.ConvTranspose2d(self.filters[i], self.filters[i+1], self.kernels[i], stride=2, padding=1))

        self.out = nn.ConvTranspose2d(self.filters[-1], out_dim[2], self.kernels[-1], stride=2, padding=1)

    def forward(self, x):
        x = self.layers[0](x)
        x = x.view(1, 128, 8, 8)
        for i in range(1, len(self.layers)):
            print(x.size())
            x = nn.functional.relu(self.layers[i](x))
        print(x.size())

            # x = nn.functional.relu(self.layers(x))
        out = nn.functional.relu(self.out(x))
        print(out.size())
        return out

'''
  Symmetrical Autoencoder.
'''
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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

def depth2pcd(depth, fovy):
    height, width = depth.shape

    # Camera intrinsics
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    cx = width / 2
    cy = height / 2

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0)

    z = np.where(valid, -depth, 0)
    x = np.where(valid, z * (c - cx) / f, 0)
    y = np.where(valid, z * (r - cy) / f, 0)
    pcd = np.dstack((x, y, z))

    return pcd.reshape(-1, 3)
    return pcd

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

    # point_cloud = np.matmul(affine_transformation, point_cloud.T)
    # point_cloud = point_cloud.T

    # Convert homogeneous to cartesian
    w = point_cloud[:, 3]
    point_cloud /= w[:, np.newaxis]

    return point_cloud[:, 0:3]


def gl2cv(depth, z_near, z_far):
    """
    Converts the depth from OpenGl to OpenCv
    :param depth: the depth in OpenGl format
    :param z_near: near clipping plane
    :param z_far: far clipping plane
    :return: a depth image
    """
    h, w = depth.shape
    linear_depth = np.zeros((h, w), dtype=np.float32)

    # for x in range(0, w):
    #     for y in range(0, h):
    #         if depth[y][x] != 1:
    #             linear_depth[y][x] = 2 * z_far * z_near / (z_far + z_near - (z_far - z_near) * (2 * depth[y][x] - 1))
    #
    # linear_depth = np.flip(linear_depth, axis=1)
    # return np.flip(linear_depth, axis=0)
    valid = np.where(depth!=1.0)
    linear_depth[valid] = 2 * z_far * z_near / (z_far + z_near - (z_far - z_near) * (2 * depth[valid] - 1))
    linear_depth = np.flip(linear_depth, axis=1)
    return np.flip(linear_depth, axis=0)


def rgb2bgr(rgb):
    """
    Converts a rgb image to bgr
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


def plot_height_map(heightmap):
    width, height = heightmap.shape
    cv_height = np.zeros((height, width), dtype=np.float32)
    min_height = np.min(heightmap)
    max_height = np.max(heightmap)
    for i in range(0, width):
        for j in range(0, height):
            cv_height[i][j] = (heightmap[i][j] - min_height) / (max_height - min_height)
    cv2.imshow("height_map", cv_height)
    cv2.waitKey()


def generate_height_map(point_cloud, shape=(100, 100), grid_step=0.0025, plot=False, rotations=0):
    """
    see kiatos19
    :param point_cloud: point cloud aligned with the target object
    :param plot: if True, plot the generated height map
    :param shape: the shape of the height map
    :param grid_step: the side of each cell in the generated height map
    :return: the height map
    """
    width = shape[0]
    height = shape[1]

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

    if rotations > 0:
        step_angle = 360 / rotations
        center = (width / 2, height / 2)
        heightmaps = []
        for i in range(rotations):
            angle = i * step_angle
            m = cv2.getRotationMatrix2D(center, angle, scale=1)
            heightmaps.append(cv2.warpAffine(height_grid, m, (height, width)))

            if plot:
                plot_height_map(heightmaps[i])

        return heightmaps
    else:
        if plot:
            plot_height_map(height_grid)

        return height_grid


def extract_features(height_map, dim, max_height, plot=False):
    """
    Extract features from height map(see kiatos19)
    :param height_map: height map aligned with the target
    :param bbox: target's dimensions
    :return: N-d feature vector
    """
    h, w = height_map.shape

    # normalize heightmap
    for i in range(0, w):
        for j in range(0, h):
            height_map[i][j] /= max_height

    m_per_pixel = 480 #ToDo:
    side = [int(m_per_pixel * dim[0]), int(m_per_pixel * dim[1])]

    # extract target mask
    mask = np.zeros((100, 100))
    for x in range(100):
        for y in range(100):
            if 50 - side[1] < x < 50 + side[1] and 50 - side[0] < y < 50 + side[0]:
                mask[x, y] = 1

    # zero pad the heightmap and mask (input dimensions power of 2)
    mask_heightmap = np.zeros((2, 128, 128))
    mask_heightmap[0, 14:114, 14:114] = height_map
    mask_heightmap[1, 14:114, 14:114] = mask

    return mask_heightmap


def draw_cell(cell, rgb):
    p1 = cell[0]
    p2 = (cell[1][0], cell[0][1])
    p3 = cell[1]
    p4 = (cell[0][0], cell[1][1])
    cv2.line(rgb, p1, p2, (0, 255, 0), thickness=1)
    cv2.line(rgb, p2, p3, (0, 255, 0), thickness=1)
    cv2.line(rgb, p3, p4, (0, 255, 0), thickness=1)
    cv2.line(rgb, p4, p1, (0, 255, 0), thickness=1)
    return rgb


def plot_point_cloud(point_cloud):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(point_cloud)
    frame = open3d.create_mesh_coordinate_frame(size=0.1)
    open3d.draw_geometries([pcd, frame])


def detect_color(rgb, color='r'):
    bgr = rgb2bgr(rgb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if color == 'r':
        # Range for lower range
        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(bgr, bgr, mask=mask)
    # cv2.imshow('mask', mask)
    # cv2.waitKey()
    return  mask