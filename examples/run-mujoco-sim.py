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

    qpos = sim.data.qpos

    for i in range(10000):

        j = sim.model.get_joint_qpos_addr('bh_wrist_joint')
        qpos[j[0]] = 0.2
        qpos[j[0]+1] = 0.54
        qpos[j[0]+2] = 0.31

        j = sim.model.get_joint_qpos_addr('bh_j11_joint')
        qpos[j] = 1

        j = sim.model.get_joint_qpos_addr('bh_j11_joint')
        qpos[j] = 1

        j = sim.model.get_joint_qpos_addr('bh_j21_joint')
        qpos[j] = 1

        sim.step()
        viewer.render()
