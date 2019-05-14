from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
import mujoco_py
import os
import math
import time

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__),
                        "../robamine/envs/assets/xml/robots/small_table_floating_bhand.xml")

    model = load_model_from_path(path)
    sim = MjSim(model)
    viewer = MjViewer(sim)


    joint_ids = [6, 7, 8, 9, 10, 11, 13, 16, 18]

    sim.step()
    init_qpos = sim.data.qpos.ravel().copy()
    init_qvel = sim.data.qvel.ravel().copy()

    for i in range(10000):
        j = sim.model.get_joint_qpos_addr('bh_wrist_joint')
        init_qpos[j[0]] = 0.2
        init_qpos[j[0]+1] = 0.54
        init_qpos[j[0]+2] = 0.31

        j = sim.model.get_joint_qpos_addr('bh_j11_joint')
        init_qpos[j] = 1
        j = sim.model.get_joint_qpos_addr('bh_j21_joint')
        init_qpos[j] = 1

        j = sim.model.get_joint_qpos_addr('bh_j22_joint')
        init_qpos[j] = 1

        old_state = sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, init_qpos, init_qvel,
                                         old_state.act, old_state.udd_state)
        sim.set_state(new_state)

        sim.step()
        viewer.render()
