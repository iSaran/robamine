#!/usr/bin/env python3
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import os

class FloatingBHand(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        """
        The constructor of the environment.

        MujocoEnv class assumes that we have defined actuators in our model, so
        this constructor does not call its parent's constructor.
        """

        path = os.path.join(os.path.dirname(__file__),
                            "../../../../autharl_core/mujoco/descriptions/xml/robots/small_table_floating_bhand.xml")

        self.model = load_model_from_path(path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # Define the action space as the Wrench in BHand wrist and torques in 4
        # actuated joints of BHand
        self.action_space = spaces.Box(low=np.array([-10, -10, -10, -2, -2, -2,
                                                      -2, -2, -2, -2]),
                                       high=np.array([10, 10, 10, 2, 2,
                                                       2, 2, 2, 2, 2]),
                                       dtype=np.float32)

        # Define the observation space as the measured Wrench in BHand wrist
        self.observation_space = spaces.Box(low=np.array([-10, -10, -10, -2, -2, -2]),
                                            high=np.array([10, 10, 10, 2, 2, 2]),
                                            dtype=np.float32)


        # Initialize this parent class because our environment wraps Mujoco's  C/C++ code.
        utils.EzPickle.__init__(self)

        # The total joint names of the BHand
        self.bhand_joint_names = ('bh_wrist_joint', 'bh_j11_joint',
                                  'bh_j12_joint', 'bh_j13_joint',
                                  'bh_j21_joint', 'bh_j22_joint',
                                  'bh_j23_joint', 'bh_j32_joint',
                                  'bh_j33_joint')
        self.bhand_joint_ids = self.get_joint_id()

        # The joints that can be actuated in BHand. The rest are passive joints
        # which are coupled with these actuated
        self.bhand_actuated_joint_names = ('bh_wrist_joint', 'bh_j11_joint',
                                           'bh_j12_joint', 'bh_j22_joint',
                                           'bh_j32_joint')
        self.bhand_actuated_joint_ids = []
        for i in self.bhand_actuated_joint_names:
            for j in self.get_joint_id(i):
                self.bhand_actuated_joint_ids.append(j)

        index = 0
        self.map = {}
        for i in self.bhand_actuated_joint_names:
            for j in self.get_joint_id(i):
                self.map[j] = index
                index = index + 1
        print(self.map)

        self.seed()


    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self.get_obs()

    def get_obs(self):
        return np.zeros(6)

    def step(self, action):
        reward = 0.0
        done = False
        obs = np.array([10, 10, 10, 10, 10, 10])
        self.do_simulation(action)
        return obs, reward, done, {}

    def do_simulation(self, action):
        bhand_wrist_joint_ids = self.get_joint_id("bh_wrist_joint")
        time = self.sim.data.time

        # Set the bias to the applied generilized force in order to
        # implement a kind of gravity compensation in bhand.
        for index in self.bhand_actuated_joint_ids:
            bias = self.sim.data.qfrc_bias[index]
            ref = bias + action[self.map[index]]
            self.sim.data.qfrc_applied[index] = ref

        # Move forward the simulation
        self.sim.step()

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

    def viewer_setup(self):
        # Set the camera configuration (spherical coordinates)
        self.viewer.cam.distance = 1.685
        self.viewer.cam.elevation = 1.9354
        self.viewer.cam.azimuth = 36.5322
