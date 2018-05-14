#!/usr/bin/env python3
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer

class FloatingBHand:
    def __init__(self, path):
        model = load_model_from_path(path)
        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.distance = 1.685
        self.viewer.cam.elevation = 1.9354
        self.viewer.cam.azimuth = 36.5322
        print('Available joint names are: {}'.format(self.sim.model.joint_names))

        # The total joint names of the BHand
        self.bhand_joint_names = ('bh_wrist_joint',
                                  'bh_j11_joint',
                                  'bh_j12_joint',
                                  'bh_j13_joint',
                                  'bh_j21_joint',
                                  'bh_j22_joint',
                                  'bh_j23_joint',
                                  'bh_j32_joint',
                                  'bh_j33_joint')

        # The joints that can be actuated in BHand. The rest are passive joints
        # which are coupled with these actuated
        self.bhand_actuated_joint_names = ('bh_wrist_joint',
                                           'bh_j11_joint',
                                           'bh_j12_joint',
                                           'bh_j22_joint',
                                           'bh_j32_joint')

    def simulate(self):
        bhand_joint_ids = self.get_joint_id()
        print(bhand_joint_ids)
        bhand_wrist_joint_ids = self.get_joint_id("bh_wrist_joint")
        print(bhand_wrist_joint_ids)
        while True:
            time = self.sim.data.time

            # Set the bias to the applied generilized force in order to
            # implement a kind of gravity compensation in bhand.
            for index in bhand_joint_ids:
                bias = self.sim.data.qfrc_bias[index]
                if index == self.get_joint_id("bh_j12_joint"):
                    cmd = 1
                else:
                    cmd = 0.0
                ref = bias + cmd
                self.sim.data.qfrc_applied[index] = ref
            print("fdfdas")
            print(self.viewer.cam.distance)
            print(self.viewer.cam.elevation)
            print(self.viewer.cam.azimuth)
            print("fdfdas")

            # Move forward the simulation
            self.sim.step()
            self.viewer.render()

    def get_joint_id(self, joint_name = "all"):
        """
        Returns a list of the indeces in mjModel.qvel that correspond to
        BHand's joints.

        Returns
        -------
        list
            The indeces (ids) of BHand's joints
        """
        output = []
        if joint_name == "all":
            for i in self.bhand_joint_names:
                index = self.sim.model.get_joint_qvel_addr(i)
                if type(index) is tuple:
                    for i in range(index[0], index[1], 1):
                        output.append(i)
                elif type(index) is np.int32:
                    output.append(index)
        else:
            index = self.sim.model.get_joint_qvel_addr(joint_name)
            if type(index) is tuple:
                for i in range(index[0], index[1], 1):
                    output.append(i)
            elif type(index) is np.int32:
                output.append(index)
        return output
