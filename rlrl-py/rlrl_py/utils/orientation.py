#!/usr/bin/env python3

import numpy as np
import numpy.matlib as mat
import numpy.linalg as lin

def quat2rot(q, shape="wxyz"):
    """
    Transforms a quaternion to a rotation matrix.
    """
    if shape == "wxyz":
        n  = q[0]
        ex = q[1]
        ey = q[2]
        ez = q[3]
    elif shape == "xyzw":
        n  = q[3]
        ex = q[0]
        ey = q[1]
        ez = q[2]
    else:
        raise RuntimeError("The shape of quaternion should be wxyz or xyzw. Given " + shape + " instead")

    R = mat.eye(3)

    R[0, 0] = 2 * (n * n + ex * ex) - 1
    R[0, 1] = 2 * (ex * ey - n * ez)
    R[0, 2] = 2 * (ex * ez + n * ey)

    R[1, 0] = 2 * (ex * ey + n * ez)
    R[1, 1] = 2 * (n * n + ey * ey) - 1
    R[1, 2] = 2 * (ey * ez - n * ex)

    R[2, 0] = 2 * (ex * ez - n * ey)
    R[2, 1] = 2 * (ey * ez + n * ex)
    R[2, 2] = 2 * (n * n + ez * ez) - 1

    return R;

def rot2quat(R, shape="wxyz"):
    """
    Transforms a rotation matrix to a quaternion.
    """

    q = [None] * 4

    tr = R[0, 0] + R[1, 1] + R[2, 2]
    print(tr)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qwh
        print("S = {}".format(S))
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S

    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
      S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
      q[0] = (R[2, 1] - R[1, 2]) / S
      q[1] = 0.25 * S
      q[2] = (R[0, 1] + R[1, 0]) / S
      q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
      S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
      q[0] = (R[0, 2] - R[2, 0]) / S
      q[1] = (R[0, 1] + R[1, 0]) / S
      q[2] = 0.25 * S
      q[3] = (R[1, 2] + R[2, 1]) / S
    else:
      S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
      q[0] = (R[1, 0] - R[0, 1]) / S
      q[1] = (R[0, 2] + R[2, 0]) / S
      q[2] = (R[1, 2] + R[2, 1]) / S
      q[3] = 0.25 * S

    return q / lin.norm(q);
