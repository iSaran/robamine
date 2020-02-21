import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from robamine.utils.math import LineSegment2D


def run():
    line_segment = LineSegment2D(np.array([0.0, 0.0]), np.array([1.0, 0.5]))
    line_segment_2 = LineSegment2D(np.array([0.2, 0.4]), np.array([1.0, 0.4]))
    print(line_segment.get_lambda([0.8, 0.4]))
    print(line_segment.get_intersection_point(line_segment_2))

    line_segment_3 = LineSegment2D(np.array([0.1, 0.3]), np.array([0.7, 0.37]))
    print('0 me 2', line_segment_3.get_intersection_point(line_segment, belong_second=True, belong_self=False))
    LineSegment2D.plot_line_segments([line_segment, line_segment_2, line_segment_3])
    print('lamva', line_segment.get_lambda([0.7521, 0.3760]))


if __name__ == '__main__':
    run()
