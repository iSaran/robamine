#!/usr/bin/env python3
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import os

import rlrl_py.utils as arl

class BHandSlidePillbox(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        """
        The constructor of the environment.

        MujocoEnv class assumes that we have defined actuators in our model, so
        this constructor does not call its parent's constructor.
        """

        path = os.path.join(os.path.dirname(__file__),
                            "assets/xml/robots/small_table_floating_bhand.xml")

        self.model = load_model_from_path(path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)

        # Initialize the Bhand joint configuration to a prespecified config
        self.init_config = {
                "bh_j11_joint": 0.0,
                "bh_j12_joint": 0.8,
                "bh_j22_joint": 0.8,
                "bh_j32_joint": 0.0
                }
        for key in self.init_config:
            self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(key)]  = self.init_config[key];
        self.sim.data.qpos[self.sim.model.get_joint_qpos_addr("bh_j21_joint")]  = self.init_config["bh_j11_joint"];
        self.sim.data.qpos[self.sim.model.get_joint_qpos_addr("bh_j13_joint")]  = 0.3333 * self.init_config["bh_j12_joint"];
        self.sim.data.qpos[self.sim.model.get_joint_qpos_addr("bh_j23_joint")]  = 0.3333 * self.init_config["bh_j22_joint"];
        self.sim.data.qpos[self.sim.model.get_joint_qpos_addr("bh_j33_joint")]  = 0.3333 * self.init_config["bh_j32_joint"];

        # Store the init state for using it for resetting the model
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # Define the action space as the two forces on the wrist of the BHand
        self.action_space = spaces.Box(low=np.array([-5, -5]),
                                       high=np.array([5,  5]),
                                       dtype=np.float32)

        # Define the observation space as the measured Wrench in BHand wrist
        self.observation_space = spaces.Box(low=np.array([-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10]),
                                            high=np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
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
        self.bhand_actuated_joint_names = ('bh_wrist_joint',
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

        # Define a map from joint actuators names to addresses in sim.data.ctrl
        # vector. This is specific for this XML model where this 4 actuators
        # are defined.
        self.joint_ctrl_addr = {
                "bh_j11_joint": 0,
                "bh_j12_joint": 1,
                "bh_j22_joint": 2,
                "bh_j32_joint": 3
                }

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self.get_obs()

    def get_obs(self):

        # Parse the joint positions
        hand_joints_pos = []
        for joint_name in self.bhand_joint_names:
            if joint_name != "bh_wrist_joint":
                hand_joints_pos.append(self.sim.data.get_joint_qpos(joint_name))

        # Parse the pose of the object
        object_pose = self.sim.data.get_joint_qpos("world_to_pillbox")
        object_wrt_world = arl.get_homogeneous_transformation(object_pose)

        # Read the wrist pose
        wrist_pose = self.sim.data.get_joint_qpos("bh_wrist_joint")
        wrist_wrt_world = arl.get_homogeneous_transformation(wrist_pose)
        wrist_wrt_object = np.linalg.inv(object_wrt_world) * wrist_wrt_world
        wrist_wrt_object_pose = arl.get_pose_from_homog(wrist_wrt_object)

        object_target_pos_wrt_world = self.sim.data.get_body_xpos("pillbox_target")
        object_target_pos_wrt_object = object_target_pos_wrt_world[0:3] - object_pose[0:3]

        return np.concatenate((wrist_wrt_object_pose, hand_joints_pos, object_target_pos_wrt_object))

    def step(self, action):
        done = False
        obs = self.get_obs()
        reward = self.get_reward(obs)
        time = self.do_simulation(action)
        if self.terminal_state(obs):
            done = True
        return obs, reward, done, {}

    def do_simulation(self, action):
        time = self.sim.data.time

        # Set the bias to the applied generilized force in order to
        # implement a kind of gravity compensation in bhand.
        index = self.get_joint_id("bh_wrist_joint")
        for i in range(0, len(index)):
            bias = self.sim.data.qfrc_bias[index[i]]
            if (i == 1 or i == 2):
                ref = bias + action[i - 1]
            else:
                ref = bias
            self.sim.data.qfrc_applied[index[i]] = ref

        # Command the joints to the initial joint configuration in order to
        # have the joints fixed in this position (no falling or external forces
        # are affecting the fingers)
        for key in self.joint_ctrl_addr:
            self.sim.data.ctrl[self.joint_ctrl_addr[key]] = self.init_config[key]

        # Move forward the simulation
        self.sim.step()
        return time

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
        self.viewer.cam.distance = 0.559478
        self.viewer.cam.elevation = -21.928236
        self.viewer.cam.azimuth = 9.089018


    def get_reward(self, observation):
        wrist_pos = observation[0:3]
        goal_pos = observation[15:18]
        reward = - 10 * np.power(np.linalg.norm(goal_pos), 2) - np.power(np.linalg.norm(wrist_pos), 2)
        return reward

    def terminal_state(self, observation):
        if np.linalg.norm(observation[0:3]) > 0.5:
            return True
        if observation[2] > 0.05:
            return True
        if np.linalg.norm(observation[15:18]) < 0.01:
            return True

        return False