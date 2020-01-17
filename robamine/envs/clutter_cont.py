"""
ClutterCont
=======

Clutter Env for continuous control
"""

import numpy as np
from numpy.linalg import norm

from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import os

from robamine.utils.robotics import PDController, Trajectory
from robamine.utils.mujoco import get_body_mass, get_body_pose, get_camera_pose, get_geom_size, get_body_inertia, get_geom_id, get_body_names
from robamine.utils.orientation import Quaternion, rot2angleaxis, rot_z, Affine3
from robamine.utils.math import sigmoid, rescale, filter_signal
import robamine.utils.cv_tools as cv_tools
import math

from robamine.utils.orientation import rot2quat

import cv2
from mujoco_py.cymj import MjRenderContext

OBSERVATION_DIM = 263

def predict_displacement_from_forces(pos_measurements, force_measurements, epsilon=1e-8, filter=0.9, outliers_cutoff=3.8, plot=False):
    import matplotlib.pyplot as plt

    # Calculate force direction
    # -------------------------
    f = force_measurements[:, :2].copy()
    f = np.nan_to_num(f / norm(f, axis=1).reshape(-1, 1))
    f_norm = np.linalg.norm(f, axis=1)

    # Find start and end of the contacts
    first = f_norm[0]
    for i in range(f_norm.shape[0]):
        if abs(f_norm[i] - first) > epsilon:
            break
    start_contact = i

    first = f_norm[-1]
    for i in reversed(range(f_norm.shape[0])):
        if abs(f_norm[i] - first) > epsilon:
            break;
    end_contact = i

    # No contact with the target detected
    if start_contact > end_contact:
        return np.zeros(2), 0.0

    f = f[start_contact:end_contact, :]

    if plot:
        fig, axs = plt.subplots(2,2)
        axs[0][0].plot(f)
        plt.title('Force')
        axs[0][1].plot(np.linalg.norm(f, axis=1))
        plt.title('norm')

    f[:,0] = filter_signal(signal=f[:,0], filter=filter, outliers_cutoff=outliers_cutoff)
    f[:,1] = filter_signal(signal=f[:,1], filter=filter, outliers_cutoff=outliers_cutoff)
    f = np.nan_to_num(f / norm(f, axis=1).reshape(-1, 1))

    if plot:
        axs[1][0].plot(f)
        plt.title('Filtered force')
        axs[1][1].plot(np.linalg.norm(f, axis=1))
        plt.title('norm')
        plt.show()

    # Velocity direction
    p = pos_measurements[start_contact:end_contact, :2].copy()
    p_dot = np.concatenate((np.zeros((1, 2)), np.diff(p, axis=0)))
    p_dot_norm = norm(p_dot, axis=1).reshape(-1, 1)
    p_dot_normalized = np.nan_to_num(p_dot / p_dot_norm)

    if plot:
        fig, axs = plt.subplots(2)
        axs[0].plot(p_dot_normalized)
        axs[0].set_title('p_dot normalized')
        axs[1].plot(p_dot)
        axs[1].set_title('p_dot')
        plt.legend(['x', 'y'])
        plt.show()

    perpedicular_to_p_dot_normalized = np.zeros(p_dot_normalized.shape)
    for i in range(p_dot_normalized.shape[0]):
        perpedicular_to_p_dot_normalized[i, :] = np.cross(np.append(p_dot_normalized[i, :], 0), np.array([0, 0, 1]))[:2]

    inner = np.diag(np.matmul(-p_dot_normalized, np.transpose(f))).copy()
    inner_perpedicular = np.diag(np.matmul(perpedicular_to_p_dot_normalized, np.transpose(f))).copy()
    if plot:
        plt.plot(inner)
        plt.title('inner product')
        plt.show()

    # Predict
    prediction = np.zeros(2)
    theta = 0.0
    for i in range(inner.shape[0]):
        prediction += p_dot_norm[i] * (inner[i] * p_dot_normalized[i, :]
                                       - inner_perpedicular[i] * perpedicular_to_p_dot_normalized[i, :])


    mean_last_inner = np.mean(inner[-10:])
    mean_last_inner = min(mean_last_inner, 1)
    mean_last_inner = max(mean_last_inner, -1)

    theta = np.sign(np.mean(inner_perpedicular[-10:])) * np.arccos(mean_last_inner)

    return prediction, theta

def get_2d_displacement(init, current):
    assert isinstance(init, Affine3)
    assert isinstance(current, Affine3)

    displacement = np.zeros(3)
    diff = init.inv() * current
    displacement[0] = diff.translation[0]
    displacement[1] = diff.translation[1]
    if np.linalg.norm(displacement) < 1e-3:
        return np.zeros(3)
    displacement[2], axis = rot2angleaxis(diff.linear)
    if axis is None:
        axis = np.array([0, 0, 1])
    displacement[2] *= np.sign(axis[2])
    return displacement

class PushingPrimitiveC:
    def __init__(self, distance = 0.1, direction_theta = 0.0, surface_size = 0.30, object_height = 0.06, finger_size = 0.02):
        self.distance = surface_size + distance
        self.direction = np.array([math.cos(direction_theta), math.sin(direction_theta)])
        self.initial_pos = - self.direction * surface_size

        # Z at the center of the target object
        # If the object is too short just put the finger right above the
        # table
        if object_height - finger_size > 0:
            offset = object_height - finger_size
        else:
            offset = 0
        self.z = finger_size + offset + 0.001

    def __str__(self):
        return "Initial Position: " + str(self.initial_pos) + "\n" + \
               "Distance: " + str(self.distance) + "\n" + \
               "Direction: " + str(self.direction) + "\n" + \
               "z: " + str(self.z) + "\n"

class Push:
    """
    Defines a push primitive action as defined in kiatos19, with the difference
    of instead of using 4 discrete directions, now we have a continuous angle
    (direction_theta) from which we calculate the direction.
    """
    def __init__(self, initial_pos = np.array([0, 0]), distance = 0.1, direction_theta = 0.0, target = True, object_height = 0.06, object_length=0.05, object_width = 0.05, finger_size = 0.02):
        self.distance = distance
        self.direction = np.array([math.cos(direction_theta), math.sin(direction_theta)])

        if target:
            # Move the target

            # Z at the center of the target object
            # If the object is too short just put the finger right above the
            # table
            if object_height - finger_size > 0:
                offset = object_height - finger_size
            else:
                offset = 0
            self.z = finger_size + offset + 0.001

            # Position outside of the object along the pushing directions.
            # Worst case scenario: the diagonal of the object
            self.initial_pos = initial_pos - (math.sqrt(pow(object_length, 2) + pow(object_width, 2)) + finger_size + 0.001) * self.direction
        else:
            self.z = 2 * object_height + finger_size + 0.001
            self.initial_pos = initial_pos

    def __str__(self):
        return "Initial Position: " + str(self.initial_pos) + "\n" + \
               "Distance: " + str(self.distance) + "\n" + \
               "Direction: " + str(self.direction) + "\n" + \
               "z: " + str(self.z) + "\n"

class PushTarget:
    def __init__(self, distance, theta, push_distance, object_height = 0.06, finger_size = 0.02):
        self.distance = distance
        self.theta = theta
        self.push_distance = push_distance

        if object_height - finger_size > 0:
            offset = object_height - finger_size
        else:
            offset = 0
        self.z = finger_size + offset + 0.001


    def get_init_pos(self):
        init = self.distance * np.array([math.cos(self.theta), math.sin(self.theta)])
        return np.append(init, self.z)

    def get_final_pos(self):
        init = self.get_init_pos()
        direction = - init / np.linalg.norm(init)
        return self.push_distance * direction

    def get_duration(self, distance_per_sec = 0.2):
        return np.linalg.norm(self.get_init_pos() - self.get_final_pos()) / distance_per_sec

class PushObstacle:
    def __init__(self, theta = 0.0, push_distance = 0.1, object_height = 0.06, finger_size = 0.02):
        self.push_distance = push_distance
        self.theta = theta
        self.z = 2 * object_height + finger_size + 0.001

    def get_init_pos(self):
        return np.array([0.0, 0.0, self.z])

    def get_final_pos(self):
        final_pos = self.push_distance * np.array([math.cos(self.theta), math.sin(self.theta)])
        return np.append(final_pos, self.z)

    def get_duration(self, distance_per_sec = 0.2):
        return np.linalg.norm(self.get_init_pos() - self.get_final_pos()) / distance_per_sec

class ClutterCont(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    The class for the Gym environment.
    """
    def __init__(self, params):
        self.params = params
        path = os.path.join(os.path.dirname(__file__),
                            "assets/xml/robots/clutter.xml")

        self.model = load_model_from_path(path)
        self.sim = MjSim(self.model)
        self._viewers = {}
        self.offscreen = MjRenderContextOffscreen(self.sim, 0)
        self.viewer = MjViewer(self.sim)

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        if self.params['discrete']:
            self.action_space = spaces.Discrete(self.params['nr_of_actions'])
        else:
            self.action_space = spaces.Box(low=np.array([-1, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)

        if self.params['extra_primitive']:
            self.nr_primitives = 3
        else:
            self.nr_primitives = 2

        obs_dimm = OBSERVATION_DIM
        if self.params['split']:
            obs_dim = int(self.params['nr_of_actions'] / self.nr_primitives) * obs_dimm
        else:
            obs_dim = obs_dimm

        self.observation_space = spaces.Box(low=np.full((obs_dim,), 0),
                                            high=np.full((obs_dim,), 0.3),
                                            dtype=np.float32)

        finger_mass = get_body_mass(self.sim.model, 'finger')
        self.pd = PDController.from_mass(mass = finger_mass)

        moment_of_inertia = get_body_inertia(self.sim.model, 'finger')
        self.pd_rot = []
        self.pd_rot.append(PDController.from_mass(mass = moment_of_inertia[0], step_response=0.005))
        self.pd_rot.append(PDController.from_mass(mass = moment_of_inertia[1], step_response=0.005))
        self.pd_rot.append(PDController.from_mass(mass = moment_of_inertia[2], step_response=0.005))

        # Parameters, updated once during reset of the model
        self.surface_normal = np.array([0, 0, 1])
        self.surface_size = np.zeros(2)  # half size of the table in x, y
        self.finger_length = 0.0
        self.finger_height = 0.0
        self.target_size = np.zeros(3)

        self.no_of_prev_points_around = 0
        # State variables. Updated after each call in self.sim_step()
        self.time = 0.0
        self.finger_pos = np.zeros(3)
        self.finger_quat = Quaternion()
        self.finger_quat_prev = Quaternion()
        self.finger_vel = np.zeros(6)
        self.finger_acc = np.zeros(3)
        self.finger_external_force_norm = 0.0
        self.finger_external_force = None
        self.target_height = 0.0
        self.target_length = 0.0
        self.target_width = 0.0
        self.target_pos = np.zeros(3)
        self.target_quat = Quaternion()
        self.push_stopped_ext_forces = False  # Flag if a push stopped due to external forces. This is read by the reward function and penalize the action
        self.last_timestamp = 0.0  # The last time stamp, used for calculating durations of time between timesteps representing experience time
        self.success = False
        self.push_distance = 0.0

        self.target_pos_prev_state = np.zeros(3)
        self.target_quat_prev_state = Quaternion()
        self.target_pos_horizon = np.zeros(3)
        self.target_quat_horizon = Quaternion()
        self.target_displacement_push_step= np.zeros(3)
        self.target_init_pose = Affine3()
        self.predicted_displacement_push_step= np.zeros(3)

        self.pos_measurements, self.force_measurements, self.target_object_displacement = None, None, None

        self.rng = np.random.RandomState()  # rng for the scene

        # Initialize this parent class because our environment wraps Mujoco's  C/C++ code.
        utils.EzPickle.__init__(self)
        self.seed()
        self.preloaded_init_state = None

    def reset_model(self):

        self.sim_step()

        if self.preloaded_init_state:
            for i in range(len(self.sim.model.geom_size)):
                self.sim.model.geom_size[i] = self.preloaded_init_state['geom_size'][i]
                self.sim.model.geom_type[i] = self.preloaded_init_state['geom_type'][i]
                self.sim.model.geom_friction[i] = self.preloaded_init_state['geom_friction'][i]
                self.sim.model.geom_condim[i] = self.preloaded_init_state['geom_condim'][i]

            # Set the initial position of the finger outside of the table, in order
            # to not occlude the objects during reading observation from the camera
            index = self.sim.model.get_joint_qpos_addr('finger')
            qpos = self.preloaded_init_state['qpos'].copy()
            qvel = self.preloaded_init_state['qvel'].copy()
            qpos[index[0]]   = 100
            qpos[index[0]+1] = 100
            qpos[index[0]+2] = 100
            self.set_state(qpos, qvel)
            self.push_distance = self.preloaded_init_state['push_distance']
            self.preloaded_init_state = None

            self.sim_step()
        else:
            random_qpos, number_of_obstacles = self.generate_random_scene()

            # Set the initial position of the finger outside of the table, in order
            # to not occlude the objects during reading observation from the camera
            index = self.sim.model.get_joint_qpos_addr('finger')
            random_qpos[index[0]]   = 100
            random_qpos[index[0]+1] = 100
            random_qpos[index[0]+2] = 100

            self.set_state(random_qpos, self.init_qvel)

            # Move forward the simulation to be sure that the objects have landed
            for _ in range(600):
                self.sim_step()

            for _ in range(300):
                for i in range(1, number_of_obstacles):
                    body_id = get_body_names(self.sim.model).index("object"+str(i))
                    self.sim.data.xfrc_applied[body_id][0] = - 3 * self.sim.data.body_xpos[body_id][0]
                    self.sim.data.xfrc_applied[body_id][1] = - 3 * self.sim.data.body_xpos[body_id][1]
                self.sim_step()

            self.check_target_occlusion(number_of_obstacles)

            for _ in range(100):
                for i in range(1, number_of_obstacles):
                     body_id = get_body_names(self.sim.model).index("object"+str(i))
                     self.sim.data.xfrc_applied[body_id][0] = 0
                     self.sim.data.xfrc_applied[body_id][1] = 0
                self.sim_step()

        # Update state variables that need to be updated only once
        self.finger_length = get_geom_size(self.sim.model, 'finger')[0]
        self.finger_height = get_geom_size(self.sim.model, 'finger')[0]  # same as length, its a sphere
        self.target_size = 2 * get_geom_size(self.sim.model, 'target')
        self.surface_size = np.array([get_geom_size(self.sim.model, 'table')[0], get_geom_size(self.sim.model, 'table')[1]])


        features, point_cloud, dim = self.get_obs()
        gap = 0.03
        points_around = []
        bbox_limit = 0.01
        for p in point_cloud:
            if (-dim[0] - bbox_limit > p[0] > -dim[0] - gap - bbox_limit or \
                    dim[0] + bbox_limit < p[0] < dim[0] + gap + bbox_limit) and \
                    -dim[1]  < p[1] < dim[1]:
                points_around.append(p)
            if (-dim[1] - bbox_limit > p[1] > -dim[1] - gap - bbox_limit or \
                    dim[1] + bbox_limit < p[1] < dim[1] + gap + bbox_limit) and \
                    -dim[0]  < p[0] < dim[0]:
                points_around.append(p)

        # cv_tools.plot_point_cloud(point_cloud)
        # cv_tools.plot_point_cloud(points_around)

        self.no_of_prev_points_around = len(points_around)
        observation, _, _ = self.get_obs()

        self.last_timestamp = self.sim.data.time
        self.success = False

        self.target_pos_prev_state = self.target_pos.copy()
        self.target_quat_prev_state = self.target_quat.copy()
        self.preloaded_init_state = None
        self.reset_horizon()
        return observation

    def reset_horizon(self):
        self.target_pos_horizon = self.target_pos.copy()
        self.target_quat_horizon = self.target_quat.copy()

    def seed(self, seed=None):
        super().seed(seed)
        self.rng.seed(seed)

    def get_obs(self):
        """
        Read depth and extract height map as observation
        :return:
        """
        self._move_finger_outside_the_table()

        # Get the depth image
        self.offscreen.render(640, 480, 0)  # TODO: xtion id is hardcoded
        rgb, depth = self.offscreen.read_pixels(640, 480, depth=True)

        z_near = 0.2 * self.sim.model.stat.extent
        z_far = 50 * self.sim.model.stat.extent
        depth = cv_tools.gl2cv(depth, z_near, z_far)

        # Generate point cloud
        fovy = self.sim.model.vis.global_.fovy
        # point_cloud = cv_tools.depth_to_point_cloud(depth, camera_intrinsics)
        point_cloud = cv_tools.depth2pcd(depth, fovy)

        # Get target pose and camera pose
        target_pose = get_body_pose(self.sim, 'target')  # g_wo: object w.r.t. world
        camera_pose = get_camera_pose(self.sim, 'xtion')  # g_wc: camera w.r.t. the world
        camera_to_target = np.matmul(np.linalg.inv(target_pose), camera_pose)  # g_oc = inv(g_wo) * g_wc

        # Transform point cloud w.r.t. to target
        point_cloud = cv_tools.transform_point_cloud(point_cloud, camera_to_target)

        # Keep the points above the table
        z = point_cloud[:, 2]
        ids = np.where((z > 0.0) & (z < 0.4))
        points_above_table = point_cloud[ids]

        dim = get_geom_size(self.sim.model, 'target')
        dim = get_geom_size(self.sim.model, 'target')
        geom_id = get_geom_id(self.sim.model, "target")
        if self.sim.model.geom_type[geom_id] == 5:
            bbox = [dim[0], dim[0], dim[1]]
        else:
            bbox = dim

        points_above_table = np.asarray(points_above_table)

        # Add the distance of the object from the edge
        distances = [self.surface_size[0] - self.target_pos[0], \
                     self.surface_size[0] + self.target_pos[0], \
                     self.surface_size[1] - self.target_pos[1], \
                     self.surface_size[1] + self.target_pos[1]]

        distances = [x / 0.5 for x in distances]

        obstacle_height_range = self.params['obstacle_height_range']
        max_height = 2 * obstacle_height_range[1]

        if self.params['split']:
            heightmaps = cv_tools.generate_height_map(points_above_table, rotations=int(self.params['nr_of_actions'] / self.nr_primitives), plot=False)
            features = []
            rot_angle = 360 / int(self.params['nr_of_actions'] / self.nr_primitives)
            for i in range(0, len(heightmaps)):
                f = cv_tools.extract_features(heightmaps[i], bbox, max_height, rotation_angle=i*rot_angle, plot=False)
                f.append(i*rot_angle)
                f.append(bbox[0] / 0.03)
                f.append(bbox[1] / 0.03)
                f.append(distances[0])
                f.append(distances[1])
                f.append(distances[2])
                f.append(distances[3])
                features.append(f)

            final_feature = np.append(features[0], features[1], axis=0)
            for i in range(2, len(features)):
                final_feature = np.append(final_feature, features[i], axis=0)
        else:
            heightmap = cv_tools.generate_height_map(points_above_table, plot=False)
            features = cv_tools.extract_features(heightmap, bbox, max_height, plot=False)
            features.append(0)
            features.append(bbox[0]/0.03)
            features.append(bbox[1]/0.03)
            features.append(distances[0])
            features.append(distances[1])
            features.append(distances[2])
            features.append(distances[3])
            final_feature = np.array(features)

        return final_feature, points_above_table, bbox

    def step(self, action):

        # Check if we are in the special case of prective horizon actions where
        # the action is an augmented action including the action index AND the
        # pose of the target object
        if isinstance(action, dict) and 'action' in action and 'pose' in action:
            _action = action['action']
            pos = np.array([action['pose'][0], action['pose'][1], 0])
            target_pos_pred = \
                  self.target_pos_horizon \
                + np.matmul(self.target_quat_horizon.rotation_matrix(), pos)
            rot = np.matmul(self.target_quat_horizon.rotation_matrix(), rot_z(action['pose'][2]))
            target_quat_pred = Quaternion.from_rotation_matrix(rot)
            pos = self.target_pos.copy()
            quat = self.target_quat.copy()
        else:
            _action = action
            pos = self.target_pos.copy()
            quat = self.target_quat.copy()

        time, self.predicted_displacement_push_step = self.do_simulation(_action, pos, quat)
        self.target_displacement_push_step = self.get_target_displacement()
        experience_time = time - self.last_timestamp
        self.last_timestamp = time
        obs, pcd, dim = self.get_obs()
        reward = self.get_reward(obs, pcd, dim, _action)

        done = False
        if self.terminal_state(obs):
            done = True

        # Extra data for having pushing distance, theta along with displacements
        # of the target
        if self.params['discrete']:
            nr_substates = (self.action_space.n / self.nr_primitives)
            j = int(_action - np.floor(_action / nr_substates) * nr_substates)
            theta = j * 2 * math.pi / nr_substates
        else:
            theta = 0.0
        extra_data = {'displacement': [self.push_distance, theta, self.target_displacement_push_step],
                      'predicted_displacement': self.predicted_displacement_push_step.copy(),
                      'push_finger_forces': self.force_measurements,
                      'push_finger_vel': self.pos_measurements,
                      'target_object_displacement': self.target_object_displacement,
                      'target_init_pose': self.target_init_pose.matrix()}


        return obs, reward, done, {'experience_time': experience_time, 'success': self.success, 'extra_data': extra_data}

    def do_simulation(self, action, pos, quat):
        primitive = int(action[0])

        # Push target primitive
        if primitive == 0:
            theta = rescale(action[1], min=-1, max=1, range=[-math.pi, math.pi])
            push_distance = rescale(action[2], min=-1, max=1, range=[self.params['push_distance'][0], self.params['push_distance'][1]])  # hardcoded read it from min max pushing distance
            distance = rescale(action[3], min=-1, max=1, range=self.params['push_target_init_distance'])  # hardcoded, read it from table limits
            push = PushTarget(distance=distance, theta=theta, push_distance=push_distance,
                              object_height = self.target_height, finger_size = self.finger_length)

        # Push obstacle primitive
        elif primitive == 1:
            theta = rescale(action[1], min=-1, max=1, range=[-math.pi, math.pi])
            push_distance = rescale(action[2], min=-1, max=1, range=[self.params['push_distance'][0], self.params['push_distance'][1]])  # hardcoded read it from min max pushing distance
            push = PushObstacle(theta=theta, push_distance=push_distance,
                                object_height = self.target_height, finger_size = self.finger_length)


        # Transform pushing from target frame to world frame
        push_initial_pos_world = np.matmul(quat.rotation_matrix(), push.get_init_pos()) + pos
        push_final_pos_world = np.matmul(quat.rotation_matrix(), push.get_final_pos()) + pos


        # Save init pose for calculating prediction of displacement w.r.t. the
        # init target pose not world
        init_target_pos = self.target_pos.copy()
        init_target_quat = self.target_quat.copy()

        init_z = 2 * self.target_height + 0.05
        self.sim.data.set_joint_qpos('finger', [push_initial_pos_world[0], push_initial_pos_world[1], init_z, 1, 0, 0, 0])
        self.sim_step()
        duration = push.get_duration()
        if isinstance(push, PushingPrimitiveC):
            duration = 3

        self.pos_measurements, self.force_measurements = None, None
        prediction = np.zeros(3)
        if self.move_joint_to_target('finger', [None, None, push.z], stop_external_forces=True)[0]:
            end = push_final_pos_world[:2]
            init_pos = self.target_pos
            _, self.pos_measurements, self.force_measurements, self.target_object_displacement = self.move_joint_to_target('finger', [end[0], end[1], None], duration)
            prediction_pos, prediction_theta = predict_displacement_from_forces(pos_measurements=self.pos_measurements, force_measurements=self.force_measurements)

            prediction[0] = prediction_pos[0]
            prediction[1] = prediction_pos[1]
            prediction[2] = prediction_theta

        else:
            self.push_stopped_ext_forces = True

        return self.sim.data.time, prediction

    def _move_finger_outside_the_table(self):
        # Move finger outside the table again
        table_size = get_geom_size(self.sim.model, 'table')
        self.sim.data.set_joint_qpos('finger', [100, 100, 100, 1, 0, 0, 0])
        self.sim_step()

    def viewer_setup(self):
        # Set the camera configuration (spherical coordinates)
        self.viewer.cam.distance = 0.75
        self.viewer.cam.elevation = -90  # default -90
        self.viewer.cam.azimuth = 90

    def get_reward(self, observation, point_cloud, dim, action):
        reward = 0.0

        # Penalize external forces during going downwards
        if self.push_stopped_ext_forces:
            return -20

        if min([observation[-4], observation[-3], observation[-2], observation[-1]]) < 0:
            return -20

        # for each push that frees the space around the target
        points_around = []
        gap = 0.03
        bbox_limit = 0.01
        for p in point_cloud:
            if (-dim[0] - bbox_limit > p[0] > -dim[0] - gap - bbox_limit or \
             dim[0] + bbox_limit < p[0] < dim[0] + gap + bbox_limit) and \
             -dim[1]  < p[1] < dim[1]:
                points_around.append(p)
            if (-dim[1] - bbox_limit > p[1] > -dim[1] - gap - bbox_limit or \
            dim[1] + bbox_limit < p[1] < dim[1] + gap + bbox_limit) and \
            -dim[0]  < p[0] < dim[0]:
                points_around.append(p)

        if self.no_of_prev_points_around == len(points_around):
            return -5

        self.no_of_prev_points_around = len(points_around)

        extra_penalty = 0
        if int(action[0]) == 0:
            extra_penalty = - rescale(action[3], min=-1, max=1, range=[0, 5])

        extra_penalty += - rescale(action[2], min=-1, max=1, range=[0, 5])

        if len(points_around) == 0:
            return +10 + extra_penalty

        return -1 + extra_penalty
        # k = max(self.no_of_prev_points_around, len(points_around))
        # if k != 0:
        #     reward = (self.no_of_prev_points_around - len(points_around)) / k
        # else:
        #     reward = 0.0
        # reward *= 10.0
        # self.no_of_prev_points_around = len(points_around)

        # cv_tools.plot_point_cloud(point_cloud)
        # cv_tools.plot_point_cloud(points_around)

        # Penalize the agent as it gets the target object closer to the edge
        # max_cost = -5
        # reward += sigmoid(observation[-1], a=max_cost, b=-15/max(self.surface_size), c=-4)
        # if observation[-1] < 0:
        #     reward = -10

        # For each object push
        # reward += -1
        # return reward

    def terminal_state(self, observation):

        # Terminal if collision is detected
        if self.push_stopped_ext_forces:
            self.push_stopped_ext_forces = False
            return True

        # Terminate if the target flips to its side, i.e. if target's z axis is
        # parallel to table, terminate.
        target_z = self.target_quat.rotation_matrix()[:,2]
        world_z = np.array([0, 0, 1])
        if np.dot(target_z, world_z) < 0.9:
            return True

        # If the object has fallen from the table
        if min([observation[-6], observation[-5], observation[-4], observation[-3]]) < 0:
            return True

        # If the object is free from obstacles around (no points around)
        if self.no_of_prev_points_around == 0:
            self.success = True
            return True

        return False

    def move_joint_to_target(self, joint_name, target_position, duration = 1, stop_external_forces=False):
        """
        Generates a trajectory in Cartesian space (x, y, z) from the current
        position of a joint to a target position. If one of the x, y, z is None
        then the joint will not move in this direction. For example:
        target_position = [None, 1, 1] will move along a trajectory in y,z and
        x will remain the same.

        TODO: The indexes of the actuators are hardcoded right now assuming
        that 0-6 is the actuator of the given joint

        Returns whether it the motion completed or stopped due to external
        forces
        """
        init_time = self.time
        desired_quat = Quaternion()
        force_measurements = np.empty((1, 3))
        pos_measurements = np.empty((1, 3))
        target_object_displacement = np.empty((1, 3))
        self.target_init_pose = Affine3.from_vec_quat(self.target_pos, self.target_quat)

        trajectory = [None, None, None]
        for i in range(3):
            if target_position[i] is None:
                target_position[i] = self.finger_pos[i]
            trajectory[i] = Trajectory([self.time, self.time + duration], [self.finger_pos[i], target_position[i]])

        while self.time <= init_time + duration:
            quat_error = self.finger_quat.error(desired_quat)

            # TODO: The indexes of the actuators are hardcoded right now
            # assuming that 0-6 is the actuator of the given joint
            for i in range(3):
                self.sim.data.ctrl[i] = self.pd.get_control(trajectory[i].pos(self.time) - self.finger_pos[i], trajectory[i].vel(self.time) - self.finger_vel[i])
                self.sim.data.ctrl[i + 3] = self.pd_rot[i].get_control(quat_error[i], - self.finger_vel[i + 3])

            self.sim_step()

            # Rotate force w.r.t init object frame
            force_wrt_target_init = np.matmul(self.target_init_pose.inv().matrix(),
                                              np.append(self.finger_external_force, 0))[0:3]
            force_measurements = np.concatenate((force_measurements, force_wrt_target_init[None, :]))

            # Rotate and translate position w.r.t init object frame
            pos_wrt_target_init = np.matmul(self.target_init_pose.inv().matrix(),
                                            np.append(self.finger_pos, 1))[0:3]
            pos_measurements = np.concatenate((pos_measurements, pos_wrt_target_init[None, :]))

            target_current_pose = Affine3.from_vec_quat(self.target_pos, self.target_quat)
            displacement = get_2d_displacement(self.target_init_pose, target_current_pose)
            displacement = displacement.reshape((1, 3))
            target_object_displacement = np.concatenate((target_object_displacement, displacement))

            current_pos = self.sim.data.get_joint_qpos(joint_name)

            if stop_external_forces and (self.finger_external_force_norm > 0.1):
                break

        # If external force is present move away
        if stop_external_forces and (self.finger_external_force_norm > 0.1):
            self.sim_step()
            # Create a new trajectory for moving the finger slightly in the
            # opposite direction to reduce the external forces
            new_trajectory = [None, None, None]
            duration = 0.2
            for i in range(3):
                direction = (target_position - self.finger_pos) / np.linalg.norm(target_position - self.finger_pos)
                new_target = self.finger_pos - 0.01 * direction  # move 1 cm backwards from your initial direction
                new_trajectory[i] = Trajectory([self.time, self.time + duration], [self.finger_pos[i], new_target[i]], [self.finger_vel[i], 0], [self.finger_acc[i], 0])

            # Perform the trajectory
            init_time = self.time
            while self.time <= init_time + duration:
                quat_error = self.finger_quat.error(desired_quat)

                # TODO: The indexes of the actuators are hardcoded right now
                # assuming that 0-6 is the actuator of the given joint
                for i in range(3):
                    self.sim.data.ctrl[i] = self.pd.get_control(new_trajectory[i].pos(self.time) - self.finger_pos[i], new_trajectory[i].vel(self.time) - self.finger_vel[i])
                    self.sim.data.ctrl[i + 3] = self.pd_rot[i].get_control(quat_error[i], - self.finger_vel[i + 3])

                self.sim_step()

            return False, pos_measurements, force_measurements, target_object_displacement

        return True, pos_measurements, force_measurements, target_object_displacement

    def sim_step(self):
        """
        A wrapper for sim.step() which updates every time a local state structure.
        """

        if self.params['render']:
            self.render()

        self.finger_quat_prev = self.finger_quat

        self.sim.step()

        self.time = self.sim.data.time

        current_pos = self.sim.data.get_joint_qpos("finger")
        self.finger_pos = np.array([current_pos[0], current_pos[1], current_pos[2]])
        self.finger_quat = Quaternion(w=current_pos[3], x=current_pos[4], y=current_pos[5], z=current_pos[6])
        if (np.inner(self.finger_quat.as_vector(), self.finger_quat_prev.as_vector()) < 0):
            self.finger_quat.w = - self.finger_quat.w
            self.finger_quat.x = - self.finger_quat.x
            self.finger_quat.y = - self.finger_quat.y
            self.finger_quat.z = - self.finger_quat.z
        self.finger_quat.normalize()

        self.finger_vel = self.sim.data.get_joint_qvel('finger')
        index = self.sim.model.get_joint_qvel_addr('finger')
        self.finger_acc = np.array([self.sim.data.qacc[index[0]], self.sim.data.qacc[index[0] + 1], self.sim.data.qacc[index[0] + 2]])

        finger_geom_id = get_geom_id(self.sim.model, "finger")
        geom2body = self.sim.model.geom_bodyid[finger_geom_id]
        self.finger_external_force_norm = np.linalg.norm(self.sim.data.cfrc_ext[geom2body])
        # functions that start with 'c' return the rotational part first, so for
        # the force take the second triplet, w.r.t. the world.
        self.finger_external_force = self.sim.data.cfrc_ext[geom2body][3:]

        # Calculate the object's length, width and height w.r.t. the surface by
        # using the orientation of the object. The height is the dimension
        # along the the surface normal. The length is the maximum dimensions
        # between the remaining two.
        dims = self.get_object_dimensions('target', self.surface_normal)
        self.target_length, self.target_width, self.target_height = dims[0], dims[1], dims[2]

        temp = self.sim.data.get_joint_qpos('target')
        self.target_pos = np.array([temp[0], temp[1], temp[2]])
        self.target_quat = Quaternion(w=temp[3], x=temp[4], y=temp[5], z=temp[6])

    def generate_random_scene(self, target_length_range=[.01, .03], target_width_range=[.01, .03],
                                    obstacle_length_range=[.01, .02], obstacle_width_range=[.01, .02],
                                    surface_length_range=[0.25, 0.25], surface_width_range=[0.25, 0.25]):
        # Randomize finger size
        geom_id = get_geom_id(self.sim.model, "finger")
        finger_height = self.rng.uniform(self.params['finger_size'][0], self.params['finger_size'][1])
        self.sim.model.geom_size[geom_id][0] = finger_height

        random_qpos = self.init_qpos.copy()

        # Randomize pushing distance
        self.push_distance = self.rng.uniform(self.params['push_distance'][0], self.params['push_distance'][1])

        # Randomize surface size
        self.surface_size[0] = self.rng.uniform(surface_length_range[0], surface_length_range[1])
        self.surface_size[1] = self.rng.uniform(surface_width_range[0], surface_width_range[1])
        geom_id = get_geom_id(self.sim.model, "table")
        self.sim.model.geom_size[geom_id][0] = self.surface_size[0]
        self.sim.model.geom_size[geom_id][1] = self.surface_size[1]


        # Randomize target object
        geom_id = get_geom_id(self.sim.model, "target")

        #   Randomize type (box or cylinder)
        temp = self.rng.uniform(0, 1)
        if (temp < self.params['target_probability_box']):
            self.sim.model.geom_type[geom_id] = 6 # id 6 is for box in mujoco
        else:
            self.sim.model.geom_type[geom_id] = 5 # id 5 is for cylinder
            # Increase the friction of the cylinders to stabilize them
            self.sim.model.geom_friction[geom_id][0] = 1.0
            self.sim.model.geom_friction[geom_id][1] = .01
            self.sim.model.geom_friction[geom_id][2] = .01
            self.sim.model.geom_condim[geom_id] = 4
            self.sim.model.geom_solref[geom_id][0] = .002

        #   Randomize size
        target_length = self.rng.uniform(target_length_range[0], target_length_range[1])
        target_width  = self.rng.uniform(target_width_range[0], min(target_length, target_width_range[1]))
        target_height = self.rng.uniform(max(self.params['target_height_range'][0], finger_height), self.params['target_height_range'][1])
        if self.sim.model.geom_type[geom_id] == 6:
            self.sim.model.geom_size[geom_id][0] = target_length
            self.sim.model.geom_size[geom_id][1] = target_width
            self.sim.model.geom_size[geom_id][2] = target_height
        elif self.sim.model.geom_type[geom_id] == 5:
            self.sim.model.geom_size[geom_id][0] = target_length
            self.sim.model.geom_size[geom_id][1] = target_height

        #   Randomize orientation
        theta = self.rng.uniform(0, 2 * math.pi)
        target_orientation = Quaternion()
        target_orientation.rot_z(theta)
        index = self.sim.model.get_joint_qpos_addr("target")
        random_qpos[index[0] + 3] = target_orientation.w
        random_qpos[index[0] + 4] = target_orientation.x
        random_qpos[index[0] + 5] = target_orientation.y
        random_qpos[index[0] + 6] = target_orientation.z

        # Randomize obstacles
        all_equal_height = self.rng.uniform(0, 1)

        if all_equal_height < self.params['all_equal_height_prob']:
            number_of_obstacles = self.params['nr_of_obstacles'][1]
        else:
            number_of_obstacles = self.params['nr_of_obstacles'][0] + self.rng.randint(self.params['nr_of_obstacles'][1] - self.params['nr_of_obstacles'][0] + 1)  # 5 to 25 obstacles

        for i in range(1, number_of_obstacles):
            geom_id = get_geom_id(self.sim.model, "object"+str(i))

            # Randomize type (box or cylinder)
            temp = self.rng.uniform(0, 1)
            if (temp < self.params['obstacle_probability_box']):
                self.sim.model.geom_type[geom_id] = 6 # id 6 is for box in mujoco
            else:
                self.sim.model.geom_type[geom_id] = 5 # id 5 is for cylinder
                # Increase the friction of the cylinders to stabilize them
                self.sim.model.geom_friction[geom_id][0] = 1.0
                self.sim.model.geom_friction[geom_id][1] = .01
                self.sim.model.geom_friction[geom_id][2] = .01
                self.sim.model.geom_condim[geom_id] = 4

            #   Randomize size
            obstacle_length = self.rng.uniform(obstacle_length_range[0], obstacle_length_range[1])
            obstacle_width  = self.rng.uniform(obstacle_width_range[0], min(obstacle_length, obstacle_width_range[1]))


            if all_equal_height < self.params['all_equal_height_prob']:
                obstacle_height = target_height
            else:
                obstacle_height = self.rng.uniform(max(self.params['obstacle_height_range'][0], finger_height), self.params['obstacle_height_range'][1])

            if self.sim.model.geom_type[geom_id] == 6:
                self.sim.model.geom_size[geom_id][0] = obstacle_length
                self.sim.model.geom_size[geom_id][1] = obstacle_width
                self.sim.model.geom_size[geom_id][2] = obstacle_height
            elif self.sim.model.geom_type[geom_id] == 5:
                self.sim.model.geom_size[geom_id][0] = obstacle_length
                self.sim.model.geom_size[geom_id][1] = obstacle_height

            # Randomize the positions
            index = self.sim.model.get_joint_qpos_addr("object" + str(i))
            r = self.rng.exponential(0.01) + target_length + max(self.sim.model.geom_size[geom_id][0], self.sim.model.geom_size[geom_id][1])
            theta = self.rng.uniform(0, 2*math.pi)
            random_qpos[index[0]] = r * math.cos(theta)
            random_qpos[index[0]+1] = r * math.sin(theta)
            random_qpos[index[0]+2] = self.sim.model.geom_size[geom_id][2]

        return random_qpos, number_of_obstacles

    def check_target_occlusion(self, number_of_obstacles):
        """
        Checks if an obstacle is above the target object and occludes it. Then
        it removes it from the arena.
        """
        for i in range(1, number_of_obstacles):
            body_id = get_body_names(self.sim.model).index("target")
            target_position = np.array([self.sim.data.body_xpos[body_id][0], self.sim.data.body_xpos[body_id][1]])
            body_id = get_body_names(self.sim.model).index("object"+str(i))
            obstacle_position = np.array([self.sim.data.body_xpos[body_id][0], self.sim.data.body_xpos[body_id][1]])

            # Continue if object has fallen off the table
            if self.sim.data.body_xpos[body_id][2] < 0:
                continue

            distance = np.linalg.norm(target_position - obstacle_position)

            target_length = self.get_object_dimensions('target', self.surface_normal)[0]
            obstacle_length = self.get_object_dimensions("object"+str(i), self.surface_normal)[0]
            if distance < 0.6 * (target_length + obstacle_length):
                index = self.sim.model.get_joint_qpos_addr("object"+str(i))
                qpos = self.sim.data.qpos.ravel().copy()
                qvel = self.sim.data.qvel.ravel().copy()
                qpos[index[0] + 2] = - 0.2
                self.set_state(qpos, qvel)

    def get_object_dimensions(self, object_name, surface_normal):
        """
        Returns the object's length, width and height w.r.t. the surface by
        using the orientation of the object. The height is the dimension
        along the the surface normal. The length is the maximum dimensions
        between the remaining two.
        """
        rot = self.sim.data.get_body_xmat(object_name)
        size = get_geom_size(self.sim.model, object_name)
        geom_id = get_geom_id(self.sim.model, object_name)
        length, width, height = 0.0, 0.0, 0.0
        if self.sim.model.geom_type[geom_id] == 6:  # if box
            if (np.abs(np.inner(rot[:, 0], surface_normal)) > 0.9):
                height = size[0]
                length = max(size[1], size[2])
                width = min(size[1], size[2])
            elif (np.abs(np.inner(rot[:, 1], surface_normal)) > 0.9):
                height = size[1]
                length = max(size[0], size[2])
                width = min(size[0], size[2])
            elif (np.abs(np.inner(rot[:, 2], surface_normal)) > 0.9):
                height = size[2]
                length = max(size[0], size[1])
                width = min(size[0], size[1])
        elif self.sim.model.geom_type[geom_id] == 5:  # if cylinder
            if (np.abs(np.inner(rot[:, 2], surface_normal)) > 0.9):
                height = size[1]
                length = size[0]
                width = size[0]
            else:
                height = size[0]
                length = size[1]
                width = size[0]
        else:
            raise RuntimeError("Object is not neither a box or a cylinder")

        return np.array([length, width, height])

    def get_target_displacement(self):
        '''Returns [x, y, theta]: the displacement of the target between pushing
        steps. Used in step for returning displacements in info.'''
        init = Affine3.from_vec_quat(self.target_pos_prev_state, self.target_quat_prev_state)
        cur = Affine3.from_vec_quat(self.target_pos, self.target_quat)
        displacement = get_2d_displacement(init, cur)
        self.target_pos_prev_state = self.target_pos.copy()
        self.target_quat_prev_state = self.target_quat.copy()
        return displacement


    def state_dict(self):
        state = {}
        state['qpos'] = self.sim.data.qpos.ravel().copy()
        state['qvel'] = self.sim.data.qvel.ravel().copy()
        state['geom_size'] = self.sim.model.geom_size.copy()
        state['geom_type'] = self.sim.model.geom_type.copy()
        state['geom_friction'] = self.sim.model.geom_friction.copy()
        state['geom_condim'] = self.sim.model.geom_condim.copy()
        state['push_distance'] = self.push_distance
        return state

    def load_state_dict(self, state):
        if state:
            self.preloaded_init_state = state.copy()
        else:
            state = None
