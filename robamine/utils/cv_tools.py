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
    return np.flip(linear_depth)


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
    cv_img = np.zeros((width, height), dtype=np.float32)
    min_value = np.min(img)
    max_value = np.max(img)
    for i in range(width):
        for j in range(height):
            cv_img[i][j] = (img[i][j] - min_value) / (max_value - min_value)
    cv2.imshow(name, cv_img)
    cv2.waitKey()


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

    def get_centroid(self, mask):
        indeces = np.argwhere(mask > 0)
        centroid = np.sum(indeces, axis=0) / float(indeces.size)
        return [int(centroid[0]), int(centroid[1])]


class Feature:
    """
    heightmap: 2d array representing the topography of the scene
    """
    def __init__(self, heightmap):
        super(Feature, self).__init__()

        self.heightmap = heightmap
        self.size = heightmap.shape
        self.center = [int(self.size[0] / 2), int(self.size[1] / 2)]

    def crop(self, x, y):
        """
        Crop the height map around the center with the given size
        """
        cropped_heightmap = self.heightmap[self.center[0] - x:self.center[0] + x,
                                           self.center[1] - y:self.center[1] + y]
        return Feature(cropped_heightmap)

    def mask_in(self, mask):
        """
        Remove from height map the pixels that do not belong to the mask
        """
        maskin_heightmap = cv2.bitwise_and(self.heightmap, self.heightmap, mask=mask)
        return Feature(maskin_heightmap)

    def mask_out(self, mask):
        """
        Remove from height map the pixels that belong to the mask
        """
        maskout_heightmap = cv2.bitwise_not(self.heightmap, self.heightmap, mask=mask)
        return Feature(maskout_heightmap)

    def pooling(self, kernel=[4, 4], stride=4, mode='AVG'):
        """
        Pooling operations on the depth and mask
        """
        out_width = int((self.size[0] - kernel[0]) / stride + 1)
        out_height = int((self.size[1] - kernel[1]) / stride + 1)

        # Perform max/avg pooling based on the mode
        pool_heightmap = np.zeros((out_width, out_height))
        for x in range(out_width):
            for y in range(out_height):
                corner = (x * stride + kernel[0], y * stride + kernel[1])
                region = [(corner[0] - kernel[0], corner[1] - kernel[1]), corner]
                if mode == 'AVG':
                    pool_heightmap[y, x] = np.sum(self.heightmap[region[0][1]:region[1][1],
                                                                 region[0][0]:region[1][0]]) / (stride * stride)
                elif mode == 'MAX':
                    pool_heightmap[y, x] = np.max(self.heightmap[region[0][1]:region[1][1],
                                                                 region[0][0]:region[1][0]])
        return Feature(pool_heightmap)

    def translate(self, tx, ty):
        """
        Translates the heightmap
        """
        t = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_heightmap = cv2.warpAffine(self.heightmap, t, self.size)
        return Feature(translated_heightmap)

    def array(self):
        """
        Return 2d array
        """
        return self.heightmap

    def flatten(self):
        """
        Flatten to 1 dim and return to use as 1dim vector
        """
        return self.heightmap.flatten()

    def plot(self, name='height_map'):
        """
        Plot the heightmap
        """
        plot_2d_img(self.heightmap, name)


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