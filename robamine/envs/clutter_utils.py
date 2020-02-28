"""
ClutterCont
=======

Clutter Env for continuous control
"""

import numpy as np
from math import cos, sin, pi, acos

from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from robamine.utils.math import LineSegment2D, triangle_area
from robamine.utils.orientation import rot_x, rot_z, rot2angleaxis
from robamine.utils.cv_tools import Feature

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TargetObjectConvexHull:
    def __init__(self, masked_in_depth):
        self.intersection = None
        self.mask_shape = masked_in_depth.shape

        self.mask_points = np.argwhere(np.transpose(masked_in_depth) > 0)


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
            plt.imsave('/tmp/convex_hull_error_mask.png', masked_in_depth, cmap='gray', vmin=np.min(masked_in_depth), vmax=np.max(masked_in_depth))
            print(err)
            raise Exception('Failed to create convex hull')

        # Calculate centroid
        self.centroid = self._get_centroid_convex_hull(hull_points)

        # Calculate height
        self.height = np.mean(masked_in_depth[masked_in_depth > 0])

        # Flags
        self.translated = False
        self.moved2world = False

    def get_limits(self, sorted=False, normalized=False):
        # Calculate the limit points
        limits = np.zeros((len(self.convex_hull), 2))
        for i in range(len(self.convex_hull)):
            limits[i, :] = self.convex_hull[i].p1.copy()

        if sorted:
            limits = limits[np.argsort(limits[:, 0])]

        if normalized:
            limits[:, 0] /= self.mask_shape[0]
            limits[:, 1] /= self.mask_shape[1]

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
        max_distance[0] = np.max(self.mask_points[:, 0])
        max_distance[1] = np.max(self.mask_points[:, 1])
        max_distance = np.linalg.norm(max_distance)
        final = max_distance * np.array([cos(theta), sin(theta)])
        push = LineSegment2D(self.centroid, self.centroid + final)

        for h in self.convex_hull:
            self.intersection = push.get_intersection_point(h)
            if self.intersection is not None:
                break

        if self.intersection is None:
            return None, None

        return self.intersection, np.linalg.norm(self.centroid - self.intersection)

    def get_pose(self):
        homog = np.eye(4)
        homog[0:2][3] = self.centroid
        return homog

    def plot(self, blocking=True):
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
        if blocking:
            plt.show()
        else:
            plt.draw()

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

