import numpy as np
from robamine.utils.orientation import Quaternion, rot2angleaxis, rot_z


def push(position, rotation_angle, push_distance=0.1):
    push_initial_pos_world = position
    push_final_pos_world = position + push_distance * np.array([np.cos(rotation_angle), np.sin(rotation_angle), 0.0])

    push_direction = (push_final_pos_world - push_initial_pos_world) / np.linalg.norm(
        push_final_pos_world - push_initial_pos_world)
    x_axis = push_direction
    y_axis = np.matmul(rot_z(np.pi / 2), x_axis)
    z_axis = np.array([0, 0, 1])
    rot_mat = np.transpose(np.array([x_axis, y_axis, z_axis]))
    push_quat = Quaternion.from_rotation_matrix(rot_mat)
    push_quat_angle, _ = rot2angleaxis(rot_mat)
    return 0


def grasp(position, rotation_angle):
    return 0