#!/usr/bin/env python3
import numpy as np
import cv2
import open3d
import math

'''
Computer Vision Utils
============
'''


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


def plot_2d_img(img, name):
    width, height = img.shape
    cv_img = np.zeros((height, width), dtype=np.float32)
    min_height = np.min(img)
    max_height = np.max(img)
    for i in range(0, width):
        for j in range(0, height):
            cv_img[i][j] = (img[i][j] - min_height) / (max_height - min_height)
    cv2.imshow(name, cv_img)
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
                plot_2d_img(heightmaps[i], 'h')

        return heightmaps
    else:
        if plot:
            plot_2d_img(height_grid, 'h')

        return height_grid


def extract_features(height_map, dim, max_height, normalize=True, rotation_angle=0, plot=False):
    """
    Extract features from height map(see kiatos19)
    :param height_map: height map aligned with the target
    :param bbox: target's dimensions
    :return: N-d feature vector
    """
    h, w = height_map.shape

    bbox = np.asarray([dim[0], dim[1]])

    if plot:
        cv_height = np.zeros((h, w), dtype=np.float32)
        min_height = np.min(height_map)
        max_height = np.max(height_map)
        for i in range(0, w):
            for j in range(0, h):
                cv_height[i][j] = ((height_map[i][j] - min_height) / (max_height - min_height))

        rgb = cv2.cvtColor(cv_height, cv2.COLOR_GRAY2RGB)

    cells = []

    cx = int(w/2)
    cy = int(h/2)

    # Target features
    m_per_pixel = 480 #ToDo:
    side = m_per_pixel * bbox

    nx, ny = [int(side[0]/4.0 + 0.5), int(side[1]/4.0 + 0.5)]

    cx1 = cx - int(side[0])
    cx2 = cx + int(side[0])
    cy1 = cy - int(side[1])
    cy2 = cy + int(side[1])
    # cells.append([(cx1, cy1), (cx, cy)])
    # cells.append([(cx, cy1), (cx2, cy)])
    # cells.append([(cx1, cy), (cx, cy2)])
    # cells.append([(cx, cy), (cx2, cy2)])

    m = cv2.getRotationMatrix2D((cx, cy), rotation_angle, scale=1)
    (cx, cy) = np.matmul(m, np.array([cx, cy, 1])).astype(int)
    c1 = np.matmul(m, np.array([cx1, cy1, 1])).astype(int)
    c2 = np.matmul(m, np.array([cx2, cy1, 1])).astype(int)
    c3 = np.matmul(m, np.array([cx2, cy2, 1])).astype(int)
    c4 = np.matmul(m, np.array([cx1, cy2, 1])).astype(int)

    cx1 = min(c1[0], c2[0], c3[0], c4[0])
    cy1 = min(c1[1], c2[1], c3[1], c4[1])
    cx2 = max(c1[0], c2[0], c3[0], c4[0])
    cy2 = max(c1[1], c2[1], c3[1], c4[1])

    #cells.append([(cx1, cy1), (cx, cy)])
    #cells.append([(cx, cy1), (cx2, cy)])
    #cells.append([(cx1, cy), (cx, cy2)])
    #cells.append([(cx, cy), (cx2, cy2)])


    # Features around target
    # 1. Define the up left corners for each 32x32 region around the target
    up_left_corners = []
    # up_left_corners.append((int(cx - 16), int(cy - side[1] - 32)))  # f_up
    # up_left_corners.append((int(cx + side[0]), int(cy - 16)))  # f_right
    # up_left_corners.append((int(cx - 16), int(cy + side[1])))  # f_down
    # up_left_corners.append((int(cx - side[0] - 32), int(cy - 16)))  # f_left

    # up_left_corners.append((int(cx - 32), int(cy - side[1] - 32)))  # f_up
    # up_left_corners.append((int(cx + side[0]), int(cy - 32)))  # f_right
    # up_left_corners.append((int(cx - 32), int(cy + side[1])))  # f_down
    # up_left_corners.append((int(cx - side[0] - 32), int(cy - 32)))  # f_left
    #
    # x_limit = [16, 8, 16, 8]
    # y_limit = [8, 16, 8, 16]
    #
    # for i in range(len(up_left_corners)):
    #     corner = up_left_corners[i]
    #     for x in range(x_limit[i]):
    #         for y in range(y_limit[i]):
    #             c = (corner[0] + x * 4, corner[1] + y * 4)
    #             cells.append([c, (c[0]+4, c[1]+4)])

    cell_resolution = 4
    dw, dh = [17, 17]

    corner = (int(cx - cell_resolution * dw / 2.0), int(cy - cell_resolution * dh / 2.0))
    for x in range(dw):
        for y in range(dh):
            c = (corner[0] + x * cell_resolution, corner[1] + y * cell_resolution)
            cells.append([c, (c[0] + cell_resolution, c[1] + cell_resolution)])

            # if 8 - nx < x < 8 + nx and 8 - ny < y < 8 + ny:
            #     mask.append(1.0)
            #     # target_cells.append([c, (c[0]+4, c[1]+4)])
            # else:
            #     mask.append(0.0)

    features = []
    for i in range(len(cells)):
        cell = cells[i]
        x1 = cell[0][0]
        x2 = cell[1][0]
        y1 = cell[0][1]
        y2 = cell[1][1]

        if i < 4:
            avg_height = np.sum(height_map[y1:y2, x1:x2]) / (side[0] * side[1])
        else:
            avg_height = np.sum(height_map[y1:y2, x1:x2]) / 16

        features.append(avg_height)

        if plot:
            rgb = draw_cell(cell, rgb)

        if plot:
            cv2.imshow('rgb', rgb)
            cv2.waitKey()

    if normalize:
        for i in range(len(features)):
            features[i] /= max_height

    return features





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


class PointCloud:
    def __init__(self):
        super(PointCloud, self).__init__()

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

    def plot_point_cloud(point_cloud):
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(point_cloud)
        frame = open3d.create_mesh_coordinate_frame(size=0.1)
        open3d.draw_geometries([pcd, frame])


color_params = {
    'red': ([0, 120, 70], [10, 255, 255]),
    'blue': (),
    'green': ()
}


class ColorDetector:
    def __init__(self, color, params=color_params):
        super(ColorDetector, self).__init__()

        self.boundaries = params[color]

    '''
    Detects the given color in a bgr image
    img: image in BGR format
    '''
    def detect(self, img, plot=False):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array(self.boundaries[0])
        upper = np.array(self.boundaries[1])

        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(hsv, lower, upper)
        if plot:
            cv2.imshow('mask', mask)
            cv2.waitKey()

        return mask


class Image2D:
    """
    A class for a 2D image with one-channel
    """
    def __init__(self, img):
        super(Image2D, self).__init__()

        self.img = img
        self.size = img.shape
        self.center = [int(self.size[0]/2), int(self.size[1]/2)]

    '''
    Computes the region of interest for the pooling operation.
    '''
    def compute_roi(self, size, cell_res):
        # if size is not given, compute the size from
        # the cell resolution
        if size is None:
            size = [int(self.size[0]/cell_res),
                    int(self.size[1]/cell_res)]
            corner = [0, 0]
        else:
            # Compute the upper left corner
            corner = [int(self.center[0] - cell_res * size[0] / 2.0),
                      int(self.center[1] - cell_res * size[1] / 2.0)]

        # Given the up left corner define the cells for average pooling
        cells = []
        for x in range(size[0]):
            for y in range(size[1]):
                c = (corner[0] + x * cell_res, corner[1] + y * cell_res)
                cells.append([c, (c[0] + cell_res, c[1] + cell_res)])
        return cells, size

    def pooling(self, size=None, cell_res=4, mode='AVG'):
        cells, size = self.compute_roi(size, cell_res)
        # Calculate the values inside each cell
        pool_img = np.zeros(size)
        for x in range(size[0]):
            for y in range(size[1]):
                cell = cells[y + x * size[1]]
                if mode == 'AVG':
                    pool_img[x, y] = np.sum(self.img[cell[0][1]:cell[1][1],
                                                     cell[0][0]:cell[1][0]]) / (cell_res * cell_res)
                elif mode == 'MAX':
                    pool_img[x, y] = np.max(self.img[cell[0][1]:cell[1][1],
                                                     cell[0][0]:cell[1][0]])
        self.img = pool_img

    def flatten(self):
        return self.img.flatten()


class RGBDHeightmap:
    def __init__(self, depth, mask, workspace=[200, 200]):
        super(RGBDHeightmap, self).__init__()

        self.w, self.h = depth.shape
        self.center = [int(self.w / 2), int(self.h / 2)]
        self.workspace = workspace
        self.mask_heightmap = mask[self.center[0] - self.workspace[0]:self.center[0] + self.workspace[0],
                                   self.center[1] - self.workspace[1]:self.center[1] + self.workspace[1]]
        self.depth_heightmap = depth[self.center[0] - self.workspace[0]:self.center[0] + self.workspace[0],
                                     self.center[1] - self.workspace[1]:self.center[1] + self.workspace[1]]


