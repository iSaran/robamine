import numpy as np
from math import cos, sin, pi
from robamine.utils.math import min_max_scale, LineSegment2D, get_centroid_convex_hull
from robamine.utils.orientation import rot_z
import matplotlib.pyplot as plt

class Feature:
    def __init__(self, name=None):
        self.name = name

    def rotate(self, angle):
        raise NotImplementedError()

    def array(self):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()

class PushAction:
    """
    A pushing action of two 3D points for init and final pos.
    """
    def __init__(self, p1, p2):
        self.p1 = p1.copy()
        self.p2 = p2.copy()

    def get_init_pos(self):
        return self.p1

    def get_final_pos(self):
        return self.p2

    def get_duration(self, distance_per_sec=0.1):
        return np.linalg.norm(self.get_init_pos() - self.get_final_pos()) / distance_per_sec

    def translate(self, p):
        self.p1 += p
        self.p2 += p
        return self

    def rotate(self, rot):
        """
        Rot: rotation matrix
        """
        self.p1 = np.matmul(rot, self.p1)
        self.p2 = np.matmul(rot, self.p2)
        return self

    def plot(self):
        raise NotImplementedError()

    def array(self):
        raise NotImplementedError()

class PushAction2D(PushAction):
    def __init__(self, p1, p2, z=0):
        super(PushAction2D, self).__init__(np.append(p1, z), np.append(p2, z))

    def translate(self, p):
        return super(PushAction2D, self).translate(np.append(p, 0))

    def rotate(self, angle):
        """
        Angle in rad.
        """
        return super(PushAction2D, self).rotate(rot_z(angle))

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        color = [0, 0, 0]
        ax.plot(self.p1[0], self.p1[1], color=color, marker='o')
        ax.plot(self.p2[0], self.p2[1], color=color, marker='.')
        ax.plot([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]], color=color, linestyle='-')
        return self


class PushTarget2D(PushAction2D):
    """
    A basic push target push with the push distance (for having the info where the target is)
    """

    def __init__(self, p1, p2, z, push_distance):
        self.push_distance = push_distance
        super(PushTarget2D, self).__init__(p1, p2, z)

    def get_target(self):
        return self.p2 - ((self.p2 - self.p1) / np.linalg.norm(self.p2 - self.p1)) * self.push_distance

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        color = [1, 0, 0]
        target = self.get_target()
        ax.plot(target[0], target[1], color=color, marker='o')

        return super(PushTarget2D, self).plot(ax)

class PushTargetWithObstacleAvoidance(PushTarget2D):
    """
    A 2D push for pushing target which uses the 2D convex hull of the object to enforce obstacle avoidance.
    convex_hull: A list of Linesegments2D. Should by in order cyclic, in order to calculate the centroid correctly
    Theta, push_distance, distance assumed to be in [-1, 1]
    """

    def __init__(self, theta, push_distance, distance, push_distance_range, init_distance_range, convex_hull,
                 object_height, finger_size):
        self.convex_hull = convex_hull  # store for plotting purposes

        # Calculate the centroid from the convex hull
        # -------------------------------------------
        hull_points = np.zeros((len(convex_hull), 2))
        for i in range(len(convex_hull)):
            hull_points[i] = convex_hull[i].p1
        centroid = get_centroid_convex_hull(hull_points)

        theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-pi, pi])

        # Calculate the initial point p1 from convex hull
        # -----------------------------------------------
        # Calculate the intersection point between the direction of the
        # push theta and the convex hull (four line segments)
        direction = np.array([cos(theta_), sin(theta_)])
        line_segment = LineSegment2D(centroid, centroid + 10 * direction)
        min_point = line_segment.get_first_intersection_point(convex_hull)
        min_point += (finger_size + 0.008) * direction
        max_point = centroid + (np.linalg.norm(centroid - min_point) + init_distance_range[1]) * direction
        distance_line_segment = LineSegment2D(min_point, max_point)
        lambd = min_max_scale(distance, range=[-1, 1], target_range=[0, 1])
        p1 = distance_line_segment.get_point(lambd)

        # Calculate the initial point p2
        # ------------------------------
        direction = np.array([cos(theta_ + pi), sin(theta_ + pi)])
        min_point = centroid
        max_point = centroid + push_distance_range[1] * direction
        push_line_segment = LineSegment2D(min_point, max_point)
        lambd = min_max_scale(push_distance, range=[-1, 1], target_range=[0, 1])
        p2 = push_line_segment.get_point(lambd)

        # Calculate height (z) of the push
        # --------------------------------
        if object_height - finger_size > 0:
            offset = object_height - finger_size
        else:
            offset = 0
        z = float(finger_size + offset + 0.001)

        super(PushTargetWithObstacleAvoidance, self).__init__(p1, p2, z, np.linalg.norm(p2 - centroid))

    def translate(self, p):
        for i in range(len(self.convex_hull)):
            self.convex_hull[i] = self.convex_hull[i].translate(p)

        return super(PushTargetWithObstacleAvoidance, self).translate(p)

    def rotate(self, angle):
        for i in range(len(self.convex_hull)):
            self.convex_hull[i] = self.convex_hull[i].rotate(angle)
        
        return super(PushTargetWithObstacleAvoidance, self).rotate(angle)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        color = [0, 0, 1]
        target = self.get_target()
        ax.plot(target[0], target[1], color=color, marker='o')
        
        for line_segment in self.convex_hull:
            ax.plot(line_segment.p1[0], line_segment.p1[1], color=color, marker='o')
            ax.plot(line_segment.p2[0], line_segment.p2[1], color=color, marker='.')
            ax.plot([line_segment.p1[0], line_segment.p2[0]], [line_segment.p1[1], line_segment.p2[1]],
                    color=color, linestyle='-')
        
        return super(PushTargetWithObstacleAvoidance, self).plot(ax)

