# # from robamine.algo.core import EvalWorld, TrainEvalWorld
# # from robamine.algo.splitddpg import SplitDDPG, Critic, Actor, ObstacleAvoidanceLoss
# from robamine import rb_logging
# import logging
import yaml
import socket
import numpy as np
import os
import pickle
import cv2

import torch
# import torch.nn as nn
# import torch.optim as optim

# # import robamine.envs.clutter_utils as clutter
# import h5py
# # from robamine.utils.memory import ReplayBuffer
# # from robamine.algo.util import OrnsteinUhlenbeckActionNoise, NormalNoise, Transition

from robamine.utils.math import min_max_scale

import robamine.utils.cv_tools as cv_tools
import robamine.envs.clutter_utils as clt_util
import robamine.algo.conv_vae as ae
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import robamine.algo.splitddpg as ddpg

import robamine.experiments.ral.combo_split_dqn as combo
from robamine.experiments.ral.supervised_push_obstacle import Actor, PushObstacleRealPolicyDeterministic, ObsDictPolicy

# logger = logging.getLogger('robamine')

PATH = '/home/iason/ral_exp'
MODELS_PATH = None

def save_img(img, name='untitled', gray=True):
    path = os.path.join(PATH, name + '.png')
    if gray:
        plt.imsave(path, img, cmap='gray', vmin=np.min(img), vmax=np.max(img))
    else:
        plt.imsave(path, img)

def draw_action(img, init_pxl, theta, mult):
    angle = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
    end_point = mult * 200 * np.array([np.cos(angle), -np.sin(angle)])
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.arrow(init_pxl[1], init_pxl[0], end_point[0], end_point[1], head_width=10, color=[1, 0, 0])
    path = os.path.join(PATH,'rgb_with_action.png')
    fig.savefig(path)


class PushTargetDepthObjectAvoidance(clt_util.PushTargetRealCartesian):
    def __init__(self, heightmap, depth, centroid_pxl_, target_pos, angle, push_distance, push_distance_range, target_height,
                 finger_length, finger_height, pixels_to_m, camera, camera_pose):
        angle_ = min_max_scale(angle, range=[-1, 1], target_range=[-np.pi, np.pi])
        push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=push_distance_range)

        # TODO: The patch has unit orientation and is the norm of the finger length. Optimally the patch should be
        # calculated using an oriented square.
        # patch_size = 2 * int(math.ceil(np.linalg.norm([finger_length, finger_length]) / pixels_to_m))
        patch_size = 35
        centroid_pxl = np.zeros(2, dtype=np.int32)
        centroid_pxl[0] = centroid_pxl_[1]
        centroid_pxl[1] = centroid_pxl_[0]
        r = 0
        step = 2
        while True:
            c = np.array([r * np.sin(-angle_), r * math.cos(-angle_)]).astype(np.int32)
            patch_center = centroid_pxl + c
            if patch_center[0] > heightmap.shape[0] or patch_center[1] > heightmap.shape[1]:
                break
            # calc patch position and extract the patch
            patch_x = int(patch_center[0] - patch_size / 2.)
            patch_y = int(patch_center[1] - patch_size / 2.)
            patch_image = heightmap[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]

            # fig, ax = plt.subplots(1)
            # ax.imshow(heightmap)
            # rect = patches.Rectangle((patch_y, patch_x), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
            # plt.show()
            # plt.imshow(patch_image)
            # plt.show()
            # print('abs', np.abs(patch_image))

            if (np.abs(patch_image) < 3e-3).all():
                z = depth[patch_center[0], patch_center[1]]
                patch_center_ = patch_center.copy()
                self.patch_center = patch_center.copy()


                self.end_point = -np.array([100 * np.cos(angle_), -100 * np.sin(angle_)])
                fig, ax = plt.subplots()
                ax.imshow(heightmap)
                ax.arrow(patch_center_[1], patch_center_[0], self.end_point[0], self.end_point[1], head_width=10, color=[1, 0, 0])
                rect = patches.Rectangle((patch_y, patch_x), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # plt.show()
                path = os.path.join(PATH,'heightmap_with_action.png')
                fig.savefig(path)


                patch_center_[0] = patch_center[1]
                patch_center_[1] = patch_center[0]
                # print('patch_center:', patch_center_)
                patch_center_image = camera.back_project(patch_center_, z)
                # print('patch_center_image:', patch_center_image)
                rgb_to_camera_frame = np.eye(3)
                patch_center_camera = np.matmul(rgb_to_camera_frame, patch_center_image)
                # patch_center_camera = patch_center_image
                patch_center__ = np.matmul(camera_pose, np.array(
                    [patch_center_camera[0], patch_center_camera[1], 0, 1.0]))[:2]
                # print('patch_center__:', patch_center__)
                x_init = patch_center__[0] - target_pos[0]
                y_init = patch_center__[1] - target_pos[1]
                break
            r += step

        super(PushTargetDepthObjectAvoidance, self).__init__(x_init=x_init, y_init=y_init, push_distance=push_distance_,
                                                             object_height=target_height, finger_size=finger_height)

class ClutterReal:
    def __init__(self, params):
        # self.camera = cv_tools.PinholeCamera(fovy, self.size)
        self.params = params.copy()
        self.camera = cv_tools.RealsenseCamera()
        self.heightmap = None
        self.heightmap_full_frame = None
        self.depth_raw = None
        self.mask_full_frame = None
        self.mask = None
        self.target_pos_pxl = None
        self.target_pos = None
        self.rgb = None

        # Surface limits in pixels
        self.surface_limits = [0.25, -0.25, 0.25, -0.25]
        self.surface_size = [0.25, 0.25]
        self.surface_limits_px = [330, 1050, 0, 720]
        self.finger_height = 0.001
        self.finger_length = 0.015
        self.pixels_to_m = 0.0012
        self.rgb_to_camera_frame = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        self.camera_pose = np.eye(4)
        self.camera_pose[:3, 3] = np.array([-0.03, -0.01, 0.66])
        self.camera_pose[:3, :3] = np.array([[1, 0, 0],
                                             [0, -1, 0],
                                             [0, 0, -1]])

        self.surface_centroid_px = [self.surface_limits_px[1] - self.surface_limits_px[0] / 2,
                                    self.surface_limits_px[3] - self.surface_limits_px[2] / 2]

        # # Load autoencoder and scaler
        vae_path = os.path.join(MODELS_PATH, 'vae')
        ae_path = os.path.join(vae_path, 'model.pkl')
        normalizer_path = os.path.join(vae_path, 'normalizer.pkl')
        with open(ae_path, 'rb') as file1:
            model = torch.load(file1, map_location='cpu')
        latent_dim = model['encoder.fc.weight'].shape[0]
        ae_params = ae.params
        ae_params['device'] = 'cpu'
        self.autoencoder = ae.ConvVae(latent_dim, ae_params)
        self.autoencoder.load_state_dict(model)
        with open(normalizer_path, 'rb') as file2:
            self.autoencoder_scaler = pickle.load(file2)

        # Load actors
        push_target_actor_path = os.path.join(MODELS_PATH, 'default/push_target.pkl')
        with open(push_target_actor_path, 'rb') as file:
            pretrained_splitddpg = pickle.load(file)
            actor = ddpg.Actor(ae.LATENT_DIM + 4, pretrained_splitddpg['action_dim'][0], [400, 300])
            actor.load_state_dict(pretrained_splitddpg['actor'][0])
        self.push_target_actor = combo.ObsDictPushTarget(actor)
        push_obstacle_actor_path = os.path.join(MODELS_PATH, 'default/push_obstacle.pkl')
        self.push_obstacle_actor = combo.ObsDictPushObstacle(Actor.load(push_obstacle_actor_path))

        self.agent = combo.SplitDQN.load(os.path.join(MODELS_PATH, 'default/combo.pkl'), self.push_target_actor, self.push_obstacle_actor,
                              0)

        self.obs_dict = {}
        self.push_init, self.push_final = None, None


    @staticmethod
    def get_obs_shapes():
        """
        Provides the shapes of the observation returned by the env. The shapes should be constants and should not change
        dynamically, because we want them to store them in arrays like h5py.
        """
        return {'finger_height': (1,),
                'finger_length': (1,),
                'heightmap_mask': (2, 386, 386),
                'surface_size': (2,),
                'target_pos': (3,),
                'target_bounding_box': (3,),
                'object_poses': (1, 7)}


    def get_obs(self):
        shapes = self.get_obs_shapes()

        obs_dict = {
            'finger_height': np.array([self.finger_height]),
            'finger_length': np.array([self.finger_length]),
            'heightmap_mask': np.zeros(shapes['heightmap_mask']),
            'surface_size': np.array(self.surface_size),
            'target_pos': np.zeros(shapes['target_pos']),
            'target_bounding_box': np.zeros(shapes['target_bounding_box']),
            'object_poses': np.zeros(shapes['object_poses']),
        }

        self.get_heightmap()
        obs_dict['heightmap_mask'][0, :] = self.heightmap
        obs_dict['heightmap_mask'][1, :] = self.mask
        obs_dict['target_pos'] = self.target_pos.copy()
        obs_dict['object_poses'][0, :3] = self.target_pos.copy()
        obs_dict['target_bounding_box'][2] = self.target_pos[2]

        obs_dict['push_target_feature'] = clt_util.get_asymmetric_actor_feature_from_dict(obs_dict, self.autoencoder,
                                                                                 self.autoencoder_scaler, angle=0,
                                                                                 primitive=0)
        obs_dict['push_obstacle_feature'] = clt_util.get_asymmetric_actor_feature_from_dict(obs_dict, self.autoencoder, None,
                                                                                   angle=0,
                                                                                   primitive=1)
        print('push target feature min max', np.min(obs_dict['push_target_feature']), np.max(obs_dict['push_target_feature']))

        print('push obstacle feature', obs_dict['push_obstacle_feature'])
        print('push obstacle feature min max', np.min(obs_dict['push_obstacle_feature']), np.max(obs_dict['push_obstacle_feature']))

        
        print('surface limits:', obs_dict['push_target_feature'][-4:])
        self.obs_dict = obs_dict.copy()
        return obs_dict

    def get_heightmap(self):
        # Load grabbed images
        with open(os.path.join(PATH, 'rgbd.pkl'), 'rb') as f:
            rgb, depth = pickle.load(f, encoding='latin1')
        
        # Calc rgb/depth blacked
        rgb[:, :self.surface_limits_px[0]] = 256
        rgb[:, self.surface_limits_px[1]:] = 256
        save_img(rgb, 'rgb_blacked')
        self.rgb = rgb.copy()
        depth[:, :self.surface_limits_px[0]] = 0
        depth[:, self.surface_limits_px[1]:] = 0
        depth[depth > 0.7] = 0.64
        depth[depth < 0.5] = 0.64
        self.depth_raw = depth.copy()
        save_img(depth, 'depth_blacked')

        # Calc mask
        color_detector = cv_tools.ColorDetector()
        mask = color_detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), color='yellow')
        save_img(mask, 'mask')
        convex_mask = np.zeros(mask.shape, dtype=np.uint8)
        target_object = clt_util.TargetObjectConvexHull(mask)
        cv2.fillPoly(convex_mask, pts=[target_object.get_limits().astype(np.int32)], color=(255, 255, 255))
        save_img(convex_mask, 'convex_mask')
        mask = convex_mask

        # Calc target pos pxl
        self.target_pos_pxl = target_object.centroid.astype(np.int32)
        # print('target_pos_pxl', self.target_pos_pxl)

        # Calc target pos wrt world
        z = depth[self.target_pos_pxl[1],  self.target_pos_pxl[0]]
        centroid_image = self.camera.back_project(self.target_pos_pxl, z)
        self.target_pos = np.matmul(self.camera_pose, np.array([centroid_image[0], centroid_image[1], centroid_image[2], 1.0]))[:3]
        self.target_pos[2] /= 2.0
        # print('target pos', self.target_pos)

        # Dilate mask due to depth inaccuracies
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=5)
        save_img(mask, 'mask_dilated')

        rgb_mask = rgb.copy()
        rgb_mask[mask == 255] = 255
        save_img(rgb_mask, 'rgb_with_mask')

        # Extract heighmap
        # plt.imshow(depth); plt.show()
        max_depth = np.max(depth)
        # print('max depth', max_depth)
        depth[depth == 0] = max_depth
        depth[depth > max_depth - 0.02] = max_depth
        heightmap = max_depth - depth
        # plt.imshow(heightmap); plt.show()
        save_img(heightmap, 'heightmap_full_frame')


        # Crop and scale heighmap and mask
        self.heightmap_full_frame = heightmap.copy()
        self.mask_full_frame = mask.copy()
        heightmap = cv_tools.Feature(heightmap).translate(self.target_pos_pxl[0], self.target_pos_pxl[1]).crop(360, 360).array()
        mask = cv_tools.Feature(mask).translate(self.target_pos_pxl[0], self.target_pos_pxl[1]).crop(360, 360).array()

        heightmap = cv2.resize(heightmap, (386, 386), interpolation = cv2.INTER_AREA)
        # plt.imshow(heightmap); plt.show()
        mask = cv2.resize(mask, (386, 386), interpolation = cv2.INTER_AREA)
        mask[mask > 0] = 255
        # mask2 = color_detector.detect(cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2BGR), color='yellow')
        save_img(heightmap, 'heightmap')
        save_img(mask, 'resized_mask')

        # # Smooth heightmap for autoencoder
        # kernel = np.ones((3, 3), np.float32) / 25
        # heightmap = cv2.filter2D(heightmap, -1, kernel)
        # plt.imshow(heightmap); plt.show()

        self.heightmap = heightmap.copy()
        # plt.imshow(self.heightmap); plt.show()
        self.mask = mask.copy()
        # self.surface_distances =
        self.target_height = self.target_pos[2] * 2
        # print('target height: ', self.target_height)

    def get_visual_representation(self, primitive):
        visual_feature = clt_util.get_actor_visual_feature(self.heightmap, self.mask, self.target_pos[2],
                                                        self.finger_height, angle=0,
                                                        primitive=primitive, plot=False)
        visual_feature = torch.FloatTensor(visual_feature).reshape(1, 1, visual_feature.shape[0],
                                                                visual_feature.shape[1]).to('cpu')
        ae_output = self.autoencoder(visual_feature).detach().cpu().numpy()[0, 0, :, :]
        visual_feature = visual_feature.detach().cpu().numpy()[0, 0, :, :]
        # print('max', np.max(ae_output))
        # print('min', np.min(ae_output))
        ae_output[ae_output >= 0.75] = 1
        ae_output[ae_output <= 0.25 ] = 0

        # ae_output[ae_output < 1 and ae_output > 0.25] = 0.5
        save_img(visual_feature, 'visual_feature' + str(primitive))
        save_img(ae_output, 'ae_output' + str(primitive))
        return visual_feature, ae_output

    def step(self):
        obs = self.get_obs()
        self.get_visual_representation(0)
        self.get_visual_representation(1)
        action = self.agent.predict(obs)
        print('action from agent:', action)

        


        if action[0] == 0:
            push = PushTargetDepthObjectAvoidance(heightmap=self.heightmap_full_frame, depth=self.depth_raw,
                                                  centroid_pxl_=self.target_pos_pxl, target_pos=self.target_pos,
                                                  angle=action[1], push_distance=action[2],
                                                  push_distance_range=self.params['push']['distance'],
                                                  finger_length=self.finger_height, finger_height=self.finger_height,
                                                  target_height=self.target_height / 2,
                                                  camera=self.camera, pixels_to_m=self.pixels_to_m,
                                                  camera_pose=self.camera_pose)
            init_pxl = push.patch_center
            draw_action(self.rgb, init_pxl, action[1], -1)
        elif action[0] == 1:
            push = clt_util.PushObstacle(theta=action[1],
                                push_distance=1,  # use maximum distance for now
                                push_distance_range=self.params['push']['distance'],
                                object_height=self.target_height,
                                finger_height=self.finger_height)
            init_pxl = np.zeros(2)
            init_pxl[0] = self.target_pos_pxl[1]
            init_pxl[1] = self.target_pos_pxl[0]
            draw_action(self.rgb, init_pxl, action[1], 1)


        push.translate(self.target_pos[:2])

        self.push_init = push.get_init_pos()
        self.push_final = push.get_final_pos()
        action_dict = {
            'init_x': self.push_init[0],
            'init_y': self.push_init[1],
            'final_x': self.push_final[0],
            'final_y': self.push_final[1],
            'height': self.push_init[2]
        }

        with open(os.path.join(PATH, 'action'), 'wb') as f:
            np.savetxt(f, np.append(self.push_init, self.push_final))

    def show(self):
        print('target_pos:', self.target_pos)
        print('target_pos_pxl:', self.target_pos_pxl)

        surface_distances = [self.obs_dict['surface_size'][0] - self.target_pos[0], \
                             self.obs_dict['surface_size'][0] + self.target_pos[0], \
                             self.obs_dict['surface_size'][1] - self.target_pos[1], \
                             self.obs_dict['surface_size'][1] + self.target_pos[1]]
        surface_distances = np.array([x / 0.5 for x in surface_distances])
        print('surface_distaces:', surface_distances)
        print('self.push_init:', self.push_init)
        print('self.push_final:', self.push_final)



if __name__ == '__main__':
    pid = os.getpid()
    print('Process ID:', pid)
    hostname = socket.gethostname()

    yml_name = 'params.yml'
    if hostname == 'dream':
        logging_dir = '/home/espa/robamine_logs/'
    elif hostname == 'triss':
        logging_dir = '/home/iason/robamine_logs/2020.01.16.split_ddpg/'
        PATH = '/home/iason/ws/ral/comms'
        MODELS_PATH = '/home/iason/ws/ral/models'
    elif hostname == 'iti-479':
        logging_dir = '/home/mkiatos/robamine/logs/'
    elif hostname == 'pc':
        logging_dir = '/home/ur5/iason/ral_ws/logs'
        PATH = '/home/ur5/iason/ral_ws/comms'
        MODELS_PATH = '/home/ur5/iason/ral_ws/models'
    else:
        raise ValueError()

    with open(yml_name, 'r') as stream:
        params = yaml.safe_load(stream)

    

    clutter = ClutterReal(params['env']['params'])
    clutter.step()
    clutter.show()
    # clutter.get_heightmap()
    # clutter.get_visual_representation(1)
    # clutter.get_action()