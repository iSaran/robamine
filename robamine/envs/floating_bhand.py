from robamine.utils.mujoco import get_body_mass, get_body_inertia, get_camera_pose, get_body_pose, get_geom_id
from robamine.utils.robotics import PDController, Trajectory
from robamine.utils.orientation import Quaternion
from robamine.utils.pcl_tools import PinholeCamera, PointCloud, gl2cv
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen, MjSimState

import numpy as np
import os
import math


class FloatingBhand:
    def __init__(self):

        self.path = "/home/mkiatos/robamine/robamine/envs/marios_assets/xml/robots/small_table_floating_bhand.xml"
        self.model = load_model_from_path(self.path)
        self.sim = MjSim(self.model)
        self.offscreen = MjRenderContextOffscreen(self.sim, 0)
        self.viewer = MjViewer(self.sim)

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

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

        # PD controllers to move the bhand
        # position controller
        mass = 1.5
        self.pd = PDController.from_mass(mass=mass)

        # orientation controller
        self.pd_rot = []
        moment_of_inertia = get_body_inertia(self.sim.model, 'bh_wrist')
        self.pd_rot.append(PDController.from_mass(mass=moment_of_inertia[0], step_response=0.05))
        self.pd_rot.append(PDController.from_mass(mass=moment_of_inertia[1], step_response=0.05))
        self.pd_rot.append(PDController.from_mass(mass=moment_of_inertia[2], step_response=0.05))

        # joints controller(different gains for fingers and spread)
        self.pd_joints = PDController(35.0, 0.0)
        self.pd_spread = PDController(30.0, 0.0)

        # Updated after each call in self.sim_step()
        self.time = 0.0
        self.bhand_pos = np.zeros(3)
        self.bhand_quat = Quaternion()
        self.bhand_quat_prev = Quaternion()
        self.bhand_vel = np.zeros(6)

        # finger joints variables
        self.bhand_joint_pos = np.zeros(4)
        self.bhand_joint_vel = np.zeros(4)

        # closing forces
        self.closing_force = 2

        # camera
        self.camera = PinholeCamera(self.sim.model.vis.global_.fovy, [640, 480])
        self.camera_pose = get_camera_pose(self.sim, 'xtion')  # g_wc
        self.rgb_to_camera_frame = np.array([[1.0, 0.0, 0.0, 0],
                                             [0.0, -1.0, 0.0, 0],
                                             [0.0, 0.0, -1.0, 0],
                                             [0.0, 0.0, 0.0, 1.0]])

        self.rng = np.random.RandomState()

        self.viewer_setup()

        self.episodes = 0
        self.success = 0

    def reset(self, seed=None):
        self.rng.seed(None)

        index = self.sim.model.get_joint_qpos_addr('bh_wrist_joint')
        self.bhand_pos = self.init_qpos[index[0]:index[0] + 3]

        init_bhand_quat = self.init_qpos[index[0] + 3:index[0] + 7]
        self.bhand_quat = Quaternion(w=init_bhand_quat[0], x=init_bhand_quat[1], \
                                     y=init_bhand_quat[2], z=init_bhand_quat[3])
        self.bhand_prev = np.zeros(6)

        self.bhand_joint_pos = np.zeros(4)
        self.bhand_joint_vel = np.zeros(4)

        random_qpos = self.init_qpos.copy()
        for i in range(1, 6):
            x = self.rng.uniform(-0.3, 0.3)
            y = self.rng.uniform(-0.3, 0.3)

            theta = self.rng.uniform(0, 2 * math.pi)
            target_orientation = Quaternion()
            target_orientation.rot_z(theta)
            index = self.sim.model.get_joint_qpos_addr("obj"+str(i))
            random_qpos[index[0] + 0] = x
            random_qpos[index[0] + 1] = y
            random_qpos[index[0] + 3] = target_orientation.w
            random_qpos[index[0] + 4] = target_orientation.x
            random_qpos[index[0] + 5] = target_orientation.y
            random_qpos[index[0] + 6] = target_orientation.z

        self.set_state(random_qpos, self.init_qvel)

    def set_state(self, qpos, qvel):
        # assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = MjSimState(old_state.time, qpos, qvel,
                               old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        for _ in range(1000):
            self.sim_step()
            self.viewer.render()

    def get_obs(self):
        # Get an rgb-d image
        self.offscreen.render(640, 480, 0)  # TODO: xtion id is hardcoded
        rgb, depth = self.offscreen.read_pixels(640, 480, depth=True)

        z_near = 0.2 * self.sim.model.stat.extent
        z_far = 50 * self.sim.model.stat.extent
        depth = gl2cv(depth, z_near, z_far)

        # Generate point cloud
        point_cloud = PointCloud.from_depth(depth, self.camera)
        point_cloud.transform(self.rgb_to_camera_frame)
        point_cloud.transform(self.camera_pose)
        return point_cloud.points

    def step(self, action):
        return 0

    def do_simulation(self, action):
        return 0

    def get_reward(self, obs, action):
        return 0

    def terminal_state(self):
        return False

    def viewer_setup(self):
        self.viewer.cam.distance = 1.0
        self.viewer.cam.elevation = -60  # default -90
        self.viewer.cam.azimuth = 90
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0

    def get_joint_id(self, joint_name="all"):
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

    def set_palm_pose(self, initial_configuration, target_pose, target_configuration, approach_direction, duration=2):
        init_time = self.time
        target_position = target_pose[0:3]
        target_quat = Quaternion(w=target_pose[3], x=target_pose[4], y=target_pose[5], z=target_pose[6])


        # Trajectory generation
        # position
        pre_grasp_position = target_position - 0.1 * approach_direction
        pre_grasp_position[2] = 0.25
        trajectory = [None, None, None]
        for i in range(3):
            trajectory[i] = Trajectory([self.time, self.time + duration], [self.bhand_pos[i], pre_grasp_position[i]])

        initial_configuration = [initial_configuration[0], initial_configuration[1] - 5 * np.pi / 180,
                                 initial_configuration[2] - 5 * np.pi / 180, initial_configuration[3] - 5 * np.pi / 180]

        # finger joints
        trajectory_joints = [None, None, None, None]
        # initial_configuration = [0.0, 38.0, 38.0, 38.0]
        for i in range(4):
            trajectory_joints[i] = Trajectory([self.time, self.time + duration], [self.bhand_joint_pos[i], initial_configuration[i]])

        while self.time <= init_time + duration:
            qpos =  self.sim.data.get_joint_qpos("bh_wrist_joint")
            qpos[3:7] = target_pose[3:7]

            # TODO: The indexes of the actuators are hardcoded right now
            # assuming that 0-6 is the actuator of the given joint
            for i in range(3):
                self.sim.data.ctrl[i] = self.pd.get_control(trajectory[i].pos(self.time) - self.bhand_pos[i], trajectory[i].vel(self.time) - self.bhand_vel[i])

            self.sim.data.ctrl[6] = self.pd_spread.get_control(trajectory_joints[0].pos(self.time) - self.bhand_joint_pos[0], trajectory_joints[0].vel(self.time) - self.bhand_joint_vel[0])
            for i in range(3):
                self.sim.data.ctrl[i+7] = self.pd_joints.get_control(trajectory_joints[i+1].pos(self.time) - self.bhand_joint_pos[i+1], trajectory_joints[i+1].vel(self.time) - self.bhand_joint_vel[i+1])

            self.sim_step()

        for i in range(3):
            trajectory[i] = Trajectory([self.time, self.time + duration], [self.bhand_pos[i], target_position[i]])

        init_time = self.time
        while self.time <= init_time + 5:
            # qpos =  self.sim.data.get_joint_qpos("bh_wrist_joint")
            # qpos[3:7] = target_pose[3:7]

            self.sim.data.ctrl[3] = self.pd_rot[0].get_control(0, - self.bhand_vel[3])
            self.sim.data.ctrl[4] = self.pd_rot[1].get_control(0, - self.bhand_vel[4])
            self.sim.data.ctrl[5] = self.pd_rot[2].get_control(0, - self.bhand_vel[5])

            # TODO: The indexes of the actuators are hardcoded right now
            # assuming that 0-6 is the actuator of the given joint
            for i in range(3):
                self.sim.data.ctrl[i] = self.pd.get_control(trajectory[i].pos(self.time) - self.bhand_pos[i], trajectory[i].vel(self.time) - self.bhand_vel[i])

            self.sim.data.ctrl[6] = self.pd_spread.get_control(trajectory_joints[0].pos(self.time) - self.bhand_joint_pos[0], trajectory_joints[0].vel(self.time) - self.bhand_joint_vel[0])
            for i in range(3):
                self.sim.data.ctrl[i+7] = self.pd_joints.get_control(trajectory_joints[i+1].pos(self.time) - self.bhand_joint_pos[i+1], trajectory_joints[i+1].vel(self.time) - self.bhand_joint_vel[i+1])

            self.sim_step()

        self.set_joint_vals(target_configuration)
        self.close_fingers()
        self.pick_the_object()
        self.pertain()
        # Open the fingers
        # self.set_joint_vals(target_configuration=np.array([0, 38.0 * np.pi / 180,
        #                                                    38.0 * np.pi / 180,
        #                                                    38.0 * np.pi / 180]), duration=5)

    def set_joint_vals(self, target_configuration, duration=4):
        init_time = self.time

        print(np.array([target_configuration]) * 180 / np.pi)

        configuration = [target_configuration[0], target_configuration[1] - 5 * np.pi / 180,
                         target_configuration[2] - 5 * np.pi / 180, target_configuration[3] - 5 * np.pi / 180]
        trajectory_joints = [None, None, None, None]
        for i in range(4):
            trajectory_joints[i] = Trajectory([self.time, self.time + duration], [self.bhand_joint_pos[i], configuration[i]])

        while self.time <= init_time + duration:
            self.sim.data.ctrl[3] = self.pd_rot[0].get_control(0, - self.bhand_vel[3])
            self.sim.data.ctrl[4] = self.pd_rot[1].get_control(0, - self.bhand_vel[4])
            self.sim.data.ctrl[5] = self.pd_rot[2].get_control(0, - self.bhand_vel[5])

            # TODO: The indexes of the actuators are hardcoded right now
            # assuming that 0-6 is the actuator of the given joint
            for i in range(3):
                self.sim.data.ctrl[i] = self.pd.get_control(0, - self.bhand_vel[i])

            self.sim.data.ctrl[6] = self.pd_spread.get_control(trajectory_joints[0].pos(self.time) - self.bhand_joint_pos[0],
                                                               trajectory_joints[0].vel(self.time) - self.bhand_joint_vel[0])
            for i in range(3):
                self.sim.data.ctrl[i+7] = self.pd_joints.get_control(trajectory_joints[i+1].pos(self.time) - self.bhand_joint_pos[i+1],
                                                                     trajectory_joints[i+1].vel(self.time) - self.bhand_joint_vel[i+1])

            ext_forces = self.sim.data.cfrc_ext.copy()
            ext_forces_norm = np.linalg.norm(ext_forces, axis=1)
            # print(ext_forces_norm)
            if (ext_forces_norm > 10).any():
                print('large force')
                break

            self.sim_step()

    def close_fingers(self, closing_force=10, duration=5):
        init_time = self.time

        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.closing_force / 2, self.closing_force / 2, self.closing_force]

        target_quat = self.bhand_quat
        target_pos = self.bhand_pos

        while self.time <= init_time + duration:

            self.sim.data.ctrl[3] = self.pd_rot[0].get_control(0, - self.bhand_vel[3])
            self.sim.data.ctrl[4] = self.pd_rot[1].get_control(0, - self.bhand_vel[4])
            self.sim.data.ctrl[5] = self.pd_rot[2].get_control(0, - self.bhand_vel[5])

            for i in range(3):
                self.sim.data.ctrl[i] = self.pd.get_control(0, - self.bhand_vel[i])

            for i in range(len(self.bhand_actuated_joint_ids)):
                index = self.bhand_actuated_joint_ids[i]
                bias = self.sim.data.qfrc_bias[index]
                ref = bias + action[i]
                self.sim.data.qfrc_applied[index] = ref

            self.sim_step()

    def pick_the_object(self, height=0.2, duration=5):
        init_time = self.time

        action = [-1.0, 1.0, 2.0, 0.0, 0.0, 0.0, self.closing_force / 2, self.closing_force / 2, self.closing_force]

        target_position = [self.bhand_pos[0], self.bhand_pos[1], self.bhand_pos[2] + height]
        target_quat = self.bhand_quat

        trajectory = [None, None, None]
        for i in range(3):
            trajectory[i] = Trajectory([self.time, self.time + duration], [self.bhand_pos[i], target_position[i]])

        while self.time <= init_time + duration:
            self.sim.data.ctrl[3] = self.pd_rot[0].get_control(0, - self.bhand_vel[3])
            self.sim.data.ctrl[4] = self.pd_rot[1].get_control(0, - self.bhand_vel[4])
            self.sim.data.ctrl[5] = self.pd_rot[2].get_control(0, - self.bhand_vel[5])

            for i in range(3):
                self.sim.data.ctrl[i] = self.pd.get_control(trajectory[i].pos(self.time) - self.bhand_pos[i], trajectory[i].vel(self.time) - self.bhand_vel[i])

            for i in range(len(self.bhand_actuated_joint_ids)):
                index = self.bhand_actuated_joint_ids[i]
                bias = self.sim.data.qfrc_bias[index]
                ref = bias + action[i]
                self.sim.data.qfrc_applied[index] = ref

            self.sim_step()

    def pertain(self, duration=5):
        init_time = self.time

        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.closing_force / 2, self.closing_force / 2, self.closing_force]

        target_position = [self.bhand_pos[0], self.bhand_pos[1], self.bhand_pos[2]]

        trajectory = [None, None, None]
        for i in range(3):
            trajectory[i] = Trajectory([self.time, self.time + duration], [self.bhand_pos[i], target_position[i]])

        while self.time <= init_time + duration:
            self.sim.data.ctrl[3] = self.pd_rot[0].get_control(0, - self.bhand_vel[3])
            self.sim.data.ctrl[4] = self.pd_rot[1].get_control(0, - self.bhand_vel[4])
            self.sim.data.ctrl[5] = self.pd_rot[2].get_control(0, - self.bhand_vel[5])

            for i in range(3):
                self.sim.data.ctrl[i] = self.pd.get_control(trajectory[i].pos(self.time) - self.bhand_pos[i], trajectory[i].vel(self.time) - self.bhand_vel[i])

            for i in range(len(self.bhand_actuated_joint_ids)):
                index = self.bhand_actuated_joint_ids[i]
                bias = self.sim.data.qfrc_bias[index]
                ref = bias + action[i]
                self.sim.data.qfrc_applied[index] = ref

            self.sim_step()

        target_obj_pose = get_body_pose(self.sim, 'obj1')
        if target_obj_pose[2, 3] > 0.15:
            self.success = 1

    def sim_step(self):
        self.bhand_quat_prev = self.bhand_quat

        self.sim.step()
        self.viewer.render()
        self.time = self.sim.data.time

        current_pos = self.sim.data.get_joint_qpos("bh_wrist_joint")
        self.bhand_pos = np.array([current_pos[0], current_pos[1], current_pos[2]])

        self.bhand_quat = Quaternion(w=current_pos[3], x=current_pos[4], y=current_pos[5], z=current_pos[6])

        if self.bhand_quat.w > 1.0:
            self.bhand_quat.w = 1.0
        elif self.bhand_quat.w < -1.0:
            self.bhand_quat.w = -1.0
        if np.inner(self.bhand_quat.as_vector(), self.bhand_quat_prev.as_vector()) < 0:
            self.bhand_quat.w = - self.bhand_quat.w
            self.bhand_quat.x = - self.bhand_quat.x
            self.bhand_quat.y = - self.bhand_quat.y
            self.bhand_quat.z = - self.bhand_quat.z
            # print('------------------------------')
        self.bhand_quat.normalize()

        self.bhand_vel = self.sim.data.get_joint_qvel('bh_wrist_joint')

        self.bhand_joint_pos[0] = self.sim.data.get_joint_qpos('bh_j11_joint')
        self.bhand_joint_pos[1] = self.sim.data.get_joint_qpos('bh_j12_joint')
        self.bhand_joint_pos[2] = self.sim.data.get_joint_qpos('bh_j22_joint')
        self.bhand_joint_pos[3] = self.sim.data.get_joint_qpos('bh_j32_joint')

        self.bhand_joint_vel[0] = self.sim.data.get_joint_qvel('bh_j11_joint')
        self.bhand_joint_vel[1] = self.sim.data.get_joint_qvel('bh_j12_joint')
        self.bhand_joint_vel[2] = self.sim.data.get_joint_qvel('bh_j22_joint')
        self.bhand_joint_vel[3] = self.sim.data.get_joint_qvel('bh_j32_joint')

        # finger_geom_id = get_geom_id(self.sim.model, "wam/bhandViz")
        # geom2body = self.sim.model.geom_bodyid[finger_geom_id]
        # self.finger_external_force_norm = np.linalg.norm(self.sim.data.cfrc_ext[geom2body])
        # print('ext_force:', self.finger_external_force_norm)

        for index in self.bhand_joint_ids:
            self.sim.data.qfrc_applied[index] = self.sim.data.qfrc_bias[index]