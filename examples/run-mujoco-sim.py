from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
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

    for i in range(10000):
        for index in joint_ids:
            sim.data.qfrc_applied[index] = sim.data.qfrc_bias[index]

        j = sim.model.get_joint_qpos_addr('bh_wrist_joint')
        sim.data.qpos[j[0]] = 0.2
        sim.data.qpos[j[0]+1] = 0.54
        sim.data.qpos[j[0]+2] = 0.31

        # j = sim.model.get_joint_qpos_addr('bh_j11_joint')
        # sim.data.qpos[j] = 0.2
        j = sim.model.get_joint_qpos_addr('bh_j21_joint')

        sim.data.qpos[j] = 0.2
        sim.step()
        viewer.render()
