#!/usr/bin/env python3
import numpy as np
import cv2
import open3d
import math
from sklearn.decomposition import PCA
from robamine.utils.orientation import rot_z, rot_x, rot2angleaxis
from math import pi
import matplotlib.pyplot as plt

'''
Computer Vision Utils
============
'''


def gl2cv(depth, z_near, z_far):
    """
    Converts the depth from OpenGl to OpenCv

    depth: the depth in OpenGl format
    z_near: near clipping plane
    z_far: far clipping plane
    return: a depth image
    """
    h, w = depth.shape
    linear_depth = np.zeros((h, w), dtype=np.float32)
    valid = np.where(depth != 1.0)
    linear_depth[valid] = 2 * z_far * z_near / (z_far + z_near - (z_far - z_near) * (2 * depth[valid] - 1))
    linear_depth = np.flip(linear_depth, axis=1)
    return np.flip(linear_depth)


def rgb2bgr(rgb):
    """
    Converts a rgb image to bgr
    (Vertical flipping of the image)
    rgb: the image in bgr format
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


class PinholeCamera:
    def __init__(self, fovy, size):
        super(PinholeCamera, self).__init__()

        self.width, self.height = size
        self.f = 0.5 * self.height / math.tan(fovy * math.pi / 360)
        self.cx = self.width / 2
        self.cy = self.height / 2

    def get_camera_matrix(self):
        camera_matrix = np.array(((self.f, 0, self.cx),
                                  (0, self.f, self.cy),
                                  (0, 0, 1)))
        return camera_matrix

    def back_project(self, p, z):
        # z /= 1000.0
        x = (p[0] - self.cx) * z / self.f
        y = (p[1] - self.cy) * z / self.f
        return np.array([x, y, z])


color_params = {
    'red': ([0, 50, 50], [20, 255, 255], [0, 0, 255]),
    'blue': ([100, 150, 0], [140, 255, 255], [255, 0, 0]),
    'green': ([40, 40, 40], [70, 255, 255], [0, 255, 0]),
    'cyan': ([80, 100, 100], [100, 255, 255], [255, 255, 0]),
    'yellow': ([20, 100, 100], [40, 255, 255], [0, 255, 255]),
    'magenta': ([140, 100, 100], [160, 255, 255], [255, 0, 255]),
    'white': ([0, 0, 230], [180, 25, 255], [255, 255, 255]),
}


class ColorDetector:
    def __init__(self, params=color_params):
        super(ColorDetector, self).__init__()
        self.params = params

    '''
    Detects the given color in a bgr image
    img: image in BGR format
    '''
    def detect(self, img, color, plot=False):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        boundaries = self.params[color]
        lower = np.array(boundaries[0])
        upper = np.array(boundaries[1])

        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(hsv, lower, upper)
        if plot:
            cv2.imshow(color, mask)
            cv2.waitKey()

        return mask

    def detect_all(self, img, plot=False):
        masked_image = np.zeros(img.shape, dtype=np.uint8)

        masks = []
        detected_colors = []
        for color in self.params:
            mask = self.detect(img, color=color)
            if (mask == 0).all():
                continue
            else:
                masks.append(mask)
                detected_colors.append(color)

            masked_image = self.apply_mask(masked_image, mask, color=self.params[color][2])

        if plot:
            cv2.imshow('masked_image', masked_image)
            cv2.waitKey()

        return masks, detected_colors

    def apply_mask(self, img, mask, color):
        """Apply the given mask to the image.
        """
        for c in range(3):
            img[:, :, c] = np.where(mask == 255, color[c], img[:, :, c])
        return img

    def get_centroid(self, mask):
        indeces = np.argwhere(mask > 0)
        centroid = np.sum(indeces, axis=0) / float(indeces.shape[0])
        return [int(centroid[0]), int(centroid[1])]

    def get_bounding_box(self, mask, plot=False):
        """
        Returns the oriented bounding box of the given mask. Returns the
        homogeneous transformation SE(2), w.r.t. the image frame along with the
        size of the bounding box.
        """
        ij = np.argwhere(mask > 0)
        xy = np.zeros(ij.shape)
        xy[:, 0] = ij[:, 1]
        xy[:, 1] = ij[:, 0]

        pca = PCA(n_components=2)
        transformed = pca.fit_transform(xy)
        homog = np.eye(4)
        homog[:2, 0] = pca.components_[0, :]
        homog[:2, 1] = pca.components_[1, :]
        if np.cross(homog[:3, 0], homog[:3, 1])[2] > 0:
            temp = homog[:, 0].copy()
            homog[:, 0] = homog[:, 1]
            homog[:, 1] = temp
        homog[2, 2] = -1

        xy_transformed = np.transpose(np.matmul(np.transpose(homog[:2, :2]), np.transpose(xy)))
        max_ = np.array([np.max(xy_transformed[:, 0]), np.max(xy_transformed[:, 1])])
        min_ = np.array([np.min(xy_transformed[:, 0]), np.min(xy_transformed[:, 1])])
        centroid = np.array([int((max_[0] + min_[0]) / 2), int((max_[1] + min_[1]) / 2)])
        bb = np.array([max_[0] - centroid[0], max_[1] - centroid[1]])
        homog[:2, 3] = np.matmul(homog[:2, :2], centroid)

        # X axis is the longer dimension of the object
        if bb[0] < bb[1]:
            homog[:3, :3] = np.matmul(rot_z(pi / 2), homog[:3, :3])
            temp = bb[0]
            bb[0] = bb[1]
            bb[1] = temp

        if plot:
            self.plot_image(mask, homog, bb)

        return homog, bb

    def plot_image(self, fig, matrix, bb):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from math import acos, pi
        _, ax = plt.subplots(1)
        ax.imshow(fig, cmap='gray', vmin=0, vmax=np.max(fig))

        # Plot image ref frame
        ax.arrow(0, 0, 25, 0, color='red', linewidth=2)
        ax.arrow(0, 0, 0, 25, color='green', linewidth=2)

        ax.arrow(matrix[0, 3], matrix[1, 3], int(matrix[0, 0] * 25), int(matrix[1, 0] * 25), color='red', linewidth=2)
        ax.arrow(matrix[0, 3], matrix[1, 3], int(matrix[0, 1] * 25), int(matrix[1, 1] * 25), color='green', linewidth=2)

        center = np.array([-bb[0], -bb[1], 0, 1])
        matrix[:3, :3] = np.matmul(matrix[:3, :3], rot_x(pi))
        center_wrt_image = np.matmul(matrix, center)
        angle, axis = rot2angleaxis(matrix[:3, :3])
        if axis is None:
            axis = np.array([0, 0, -1])
        rect = patches.Rectangle((center_wrt_image[0], center_wrt_image[1]), 2*bb[0], 2*bb[1], angle=axis[2] * (180/pi)*angle, linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.show()

    def get_height(self, depth, mask):
        """
        Returns the average value of the masked region
        """
        indices = np.argwhere(mask > 0)
        height = 0
        for id in indices:
            height += depth[id[0]][id[1]]
        return height / indices.shape[0]



class Feature:
    """
    heightmap: 2d array representing the topography of the scene
    """
    def __init__(self, heightmap):
        super(Feature, self).__init__()

        self.heightmap = heightmap
        self.size = heightmap.shape
        self.center = [int(self.size[0] / 2), int(self.size[1] / 2)]

    def increase_canvas_size(self, x, y):
        assert x >= self.size[0] and y >= self.size[1]
        new_center = [int(x / 2), int(y / 2)]
        if len(self.heightmap.shape) == 3:
            new_shape = (x, y, self.heightmap.shape[2])
        else:
            new_shape = (x, y)
        new_canvas = np.zeros(new_shape, dtype=self.heightmap.dtype)
        first_row = new_center[0] - self.center[0]
        first_col = new_center[1] - self.center[1]
        last_row = first_row + self.size[0]
        last_col = first_col + self.size[1]
        new_canvas[first_row:last_row, first_col:last_col] = self.heightmap.copy()
        return Feature(new_canvas)

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
        mask = mask.astype(np.int8)
        maskin_heightmap = cv2.bitwise_and(self.heightmap, self.heightmap, mask=mask)
        return Feature(maskin_heightmap)

    def mask_out(self, mask):
        """
        Remove from height map the pixels that belong to the mask
        """
        mask = (255 - mask).astype(np.int8)
        maskout_heightmap = cv2.bitwise_and(self.heightmap, self.heightmap, mask=mask)
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
        tx = self.center[0] - tx
        ty = self.center[1] - ty
        t = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_heightmap = cv2.warpAffine(self.heightmap, t, self.size)
        return Feature(translated_heightmap)

    def rotate(self, theta):
        """
        Rotate the heightmap around its center

        theta: Rotation angle in degrees. Positive values mean counter-clockwise rotation .
        """
        scale = 1.0
        rot = cv2.getRotationMatrix2D((self.center[0], self.center[1]), theta, scale)
        rotated_heightmap = cv2.warpAffine(self.heightmap, rot, self.size)
        return Feature(rotated_heightmap)

    def non_zero_pixels(self):
        return np.argwhere(self.heightmap > 0).shape[0]

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

    def plot(self, name='height_map', grayscale=True):
        """
        Plot the heightmap
        """
        if grayscale:
            plt.imshow(self.heightmap, cmap='gray', vmin=np.min(self.heightmap), vmax=np.max(self.heightmap))
        else:
            plt.imshow(self.heightmap)
        plt.title(name)
        plt.show()
        return self

    def normalize(self, max_height):
        normalized = self.heightmap / max_height
        normalized[normalized > 1] = 1
        normalized[normalized < 0] = 0
        return Feature(normalized)

def get_circle_mask(x, y, radius, inverse=False):
    circle_mask = np.zeros((x, y))
    center = [int(x / 2), int(y / 2)]
    for i in range(x):
        for j in range(y):
            if np.linalg.norm([i - center[0], j - center[1]]) < radius:
                circle_mask[i, j] = 256
    if inverse:
        circle_mask = cv2.bitwise_not(circle_mask)
    return circle_mask


class PointCloud:
    def __init__(self):
        super(PointCloud, self).__init__()
        self.points = []

    @classmethod
    def from_depth(cls, depth, camera):
        cls = PointCloud()
        depth = depth
        width, height = depth.shape
        c, r = np.meshgrid(np.arange(height), np.arange(width), sparse=True)
        valid = (depth > 0)
        z = np.where(valid, depth, 0)
        x = np.where(valid, z * (c - camera.cx) / camera.f, 0)
        y = np.where(valid, z * (r - camera.cy) / camera.f, 0)
        pcd = np.dstack((x, y, z))
        cls.points = pcd.reshape(-1, 3)
        return cls

    def transform(self, tr_matrix):
        """
        Applies an affine transformation to the point cloud
        tr_matrix: 4x4 matrix that describes the affine transformation [R|t]
        """
        # Convert cartesian to homogeneous coordinates
        ones = np.ones((self.points.shape[0], 1), dtype=np.float32)
        self.points = np.concatenate((self.points, ones), axis=1)

        # Transform cloud
        for i in range(self.points.shape[0]):
            self.points[i] = np.matmul(tr_matrix, self.points[i])

        # Convert homogeneous to cartesian
        w = self.points[:, 3]
        self.points /= w[:, np.newaxis]
        self.points = self.points[:, 0:3]

    def size(self):
        return self.points.shape[0]

    def generate_height_map(self, size=(100, 100), grid_step=0.0025, rotations=0, plot=False):
        """
        see kiatos19
        Point_cloud must be point cloud aligned with the target object
        plot: if True, plot the generated height map
        size: the shape of the height map
        grid_step: the side of each cell in the generated height map
        :return: the height map
        """
        width, height = size

        height_grid = np.zeros((height, width), dtype=np.float32)

        for i in range(self.points.shape[0]):
            x = self.points[i][0]
            y = self.points[i][1]
            z = self.points[i][2]

            idx_x = int(np.floor(x / grid_step)) + int(width / 2)
            idx_y = int(np.floor(y / grid_step)) + int(height / 2)

            if 0 < idx_x < width - 1 and 0 < idx_y < height - 1:
                if height_grid[height - idx_y][idx_x] < z:
                    height_grid[height - idx_y][idx_x] = z

        if rotations > 0:
            step_angle = 360 / rotations
            center = (width / 2, height / 2)
            heightmaps = []
            for i in range(rotations):
                angle = i * step_angle
                m = cv2.getRotationMatrix2D(center, -angle, scale=1)
                heightmaps.append(cv2.warpAffine(height_grid, m, (height, width)))
                if plot:
                    plot_2d_img(heightmaps[i], 'height_map' + str(i))
            return heightmaps
        else:
            if plot:
                plot_2d_img(height_grid, 'height_map')

        return height_grid

    def plot(self):
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(self.points)
        frame = open3d.create_mesh_coordinate_frame(size=0.1)
        open3d.draw_geometries([pcd, frame])

    def crop(self, min, max, axis=0):
        ids = np.where((self.points[:, axis] > min) & (self.points[:, axis] < max))
        self.points = self.points[ids]

