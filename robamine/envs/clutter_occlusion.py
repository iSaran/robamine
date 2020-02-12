"""
ClutterOcclusion
================

Clutter Env for exploring scenes of occluded targets
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

OBSERVATION_DIM = 264

class Push:
    pass
    # Push x,y,z -> r,theta
    # iason

class ClutterOcclusion(mujoco_env.MujocoEnv, utils.EzPickle):
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

        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        obs_dim = OBSERVATION_DIM

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

    # Basic API methods called during running the env
    # -----------------------------------------------

    def reset_model(self):
        self.sim_step()

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

        for _ in range(100):
            for i in range(1, number_of_obstacles):
                 body_id = get_body_names(self.sim.model).index("object"+str(i))
                 self.sim.data.xfrc_applied[body_id][0] = 0
                 self.sim.data.xfrc_applied[body_id][1] = 0
            self.sim_step()

        return np.zeros(OBSERVATION_DIM)

    def step(self, action):
        time = self.do_simulation(action)
        obs = self.get_obs()
        reward = self.get_reward(obs, action)
        done = False
        if self.terminal_state(obs):
            done = True
        return obs, reward, done, {'success': self.success}

    def seed(self, seed=None):
        super().seed(seed)
        self.rng.seed(seed)


    # MDP methods (state, actions, reward, terminal states etc)
    # ---------------------------------------------------------

    def get_obs(self):
        self._move_finger_outside_the_table()

        self.offscreen.render(640, 480, 0)  # TODO: xtion id is hardcoded
        rgb, depth = self.offscreen.read_pixels(640, 480, depth=True)

        z_near = 0.2 * self.sim.model.stat.extent
        z_far = 50 * self.sim.model.stat.extent
        depth = cv_tools.gl2cv(depth, z_near, z_far)

        bgr = cv_tools.rgb2bgr(rgb)
        color_detector = cv_tools.ColorDetector('red')
        mask = color_detector.detect(bgr)

        # Pre-process rgb-d height maps
        workspace = [190, 190]
        center = [240, 320]
        depth = depth[center[0] - workspace[0]:center[0] + workspace[0],
                      center[1] - workspace[1]:center[1] + workspace[1]]
        max_depth = np.max(depth)
        depth = max_depth - depth
        mask = mask[center[0] - workspace[0]:center[0] + workspace[0],
                    center[1] - workspace[1]:center[1] + workspace[1]]

        centroid = color_detector.get_centroid(mask)

        cv_tools.plot_2d_img(depth, 'depth')
        cv_tools.plot_2d_img(mask, 'mask')
        # ToDo: normalize depth??

        depth_feature = cv_tools.Feature(depth)
        depth_feature.translate(centroid[1], centroid[0]).plot()

        return None

    def do_simulation(self, action):
        # Receive action from ddpg and create push
        # iason
        return self.sim.data.time

    def get_reward(self, observation, action):
            # - reward
            #   - diff btwn pixels of target
            #   - scene changed
            #   - object revealed
            #   - collision -50
            #   - penalize long pushes
            #   - guide exploration to pass through large heights
        reward = 0.0
        return reward

    def terminal_state(self, observation):
        # decect object fully (?)

        return False


    # Auxialliary methods
    # -------------------
    def _move_finger_outside_the_table(self):
        # Move finger outside the table again
        table_size = get_geom_size(self.sim.model, 'table')
        self.sim.data.set_joint_qpos('finger', [100, 100, 100, 1, 0, 0, 0])
        self.sim_step()

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

    def viewer_setup(self):
        # Set the camera configuration (spherical coordinates)
        self.viewer.cam.distance = 0.75
        self.viewer.cam.elevation = -90  # default -90
        self.viewer.cam.azimuth = 90

    def generate_random_scene(self, target_length_range=[.0125, .02], target_width_range=[.0125, .02],
                                    obstacle_length_range=[.01, .02], obstacle_width_range=[.01, .02],
                                    surface_length_range=[0.25, 0.25], surface_width_range=[0.25, 0.25]):
        return random_qpos, number_of_obstacles

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
        
    def generate_random_scene(self, target_length_range=[.0125, .02], target_width_range=[.0125, .02],
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
