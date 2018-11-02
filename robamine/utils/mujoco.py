#!/usr/bin/env python3

from mujoco_py.generated import const

'''
Mujoco Utils
============

Utilities wrapper functions for MuJoCo-Py

'''

def set_mocap_pose(sim, pos, quat):
    ''' Sets the pose (position and orientation) of a mocap body as a relative
    pose w.r.t. the current.

    Arguments
    ---------
        - sim: The MjSim object
        - pos: The position as a 3x1 vector
        - quat: The orientation as a quaternion (4x1 vector: w, x, y, z)
    '''
    assert sim.model.nmocap != 0

    reset_mocap2body_xpos(sim)
    sim.data.mocap_pos[:] = sim.data.mocap_pos + pos

def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]
