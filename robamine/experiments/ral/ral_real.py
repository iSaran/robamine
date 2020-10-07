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

# logger = logging.getLogger('robamine')

PATH = '/home/iason/ral_exp'
VAE_PATH = None

def save_img(img, name='untitled', gray=True):
    path = os.path.join(PATH, name + '.png')
    if gray:
        plt.imsave(path, img, cmap='gray', vmin=np.min(img), vmax=np.max(img))
    else:
        plt.imsave(path, img)

class PushTargetDepthObjectAvoidance(clt_util.PushTargetRealCartesian):
    def __init__(self, raw_depth, centroid_pxl_, angle, push_distance, push_distance_range, target_height,
                 finger_length, finger_height, pixels_to_m, camera, rgb_to_camera_frame, camera_pose):
        angle_ = min_max_scale(angle, range=[-1, 1], target_range=[-np.pi, np.pi])
        push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=push_distance_range)

        raw_depth_ = raw_depth
        table_depth = np.max(raw_depth_)

        # TODO: The patch has unit orientation and is the norm of the finger length. Optimally the patch should be
        # calculated using an oriented square.
        patch_size = 2 * int(math.ceil(np.linalg.norm([finger_length, finger_length]) / pixels_to_m))
        centroid_pxl = np.zeros(2, dtype=np.int32)
        centroid_pxl[0] = centroid_pxl_[1]
        centroid_pxl[1] = centroid_pxl_[0]
        r = 0
        step = 2
        while True:
            c = np.array([r * np.sin(-angle_), r * math.cos(-angle_)]).astype(np.int32)
            patch_center = centroid_pxl + c
            if patch_center[0] > raw_depth_.shape[0] or patch_center[1] > raw_depth_.shape[1]:
                break
            # calc patch position and extract the patch
            patch_x = int(patch_center[0] - patch_size / 2.)
            patch_y = int(patch_center[1] - patch_size / 2.)
            patch_image = raw_depth_[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]

            fig, ax = plt.subplots(1)
            ax.imshow(raw_depth_)
            rect = patches.Rectangle((patch_y, patch_x), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.show()
            plt.imshow(patch_image)
            plt.show()
            print('abs', np.abs(patch_image - table_depth))
            print('table', table_depth)

            if (np.abs(patch_image - table_depth) < 1e-3).all():
                z = raw_depth_[patch_center[0], patch_center[1]]
                patch_center_ = patch_center.copy()
                patch_center_[0] = patch_center[1]
                patch_center_[1] = patch_center[0]
                patch_center_image = camera.back_project(patch_center_, z)
                patch_center_camera = np.matmul(rgb_to_camera_frame, patch_center_image)
                # patch_center_camera = patch_center_image
                patch_center__ = np.matmul(camera_pose, np.array(
                    [patch_center_camera[0], patch_center_camera[1], 0, 1.0]))[:2]
                x_init = patch_center__[0] - self.target_pos[0]
                y_init = patch_center__[1] - self.target_pos[1]
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
        self.mask = None
        self.target_pos_pxl = None
        self.target_pos = None

        # Surface limits in pixels
        self.surface_limits = [0.25, -0.25, 0.25, -0.25]
        self.surface_limits_px = [330, 1050, 0, 720]
        self.finger_height = 0.001
        self.finger_length = 0.015
        self.pixels_to_m = 0.0012
        self.surface_centroid_px = [self.surface_limits_px[1] - self.surface_limits_px[0] / 2,
                                    self.surface_limits_px[3] - self.surface_limits_px[2] / 2]

        # # Load autoencoder and scaler
        ae_path = os.path.join(VAE_PATH, 'model.pkl')
        normalizer_path = os.path.join(VAE_PATH, 'normalizer.pkl')
        with open(ae_path, 'rb') as file1:
            model = torch.load(file1, map_location='cpu')
        latent_dim = model['encoder.fc.weight'].shape[0]
        ae_params = ae.params
        ae_params['device'] = 'cpu'
        self.autoencoder = ae.ConvVae(latent_dim, ae_params)
        self.autoencoder.load_state_dict(model)
        with open(normalizer_path, 'rb') as file2:
            self.autoencoder_scaler = pickle.load(file2)

    def get_heightmap(self):
        # Load grabbed images
        with open(os.path.join(PATH, 'rgbd.pkl'), 'rb') as f:
            rgb, depth = pickle.load(f, encoding='latin1')
        
        rgb[:, :self.surface_limits_px[0]] = 256
        rgb[:, self.surface_limits_px[1]:] = 256
        save_img(rgb, 'rgb_blacked')

        depth[:, :self.surface_limits_px[0]] = 0
        depth[:, self.surface_limits_px[1]:] = 0
        save_img(depth, 'depth_blacked')
        print('depth: min:', np.min(depth), 'max:', np.max(depth))

        rgb_cropped = rgb[:, self.surface_limits_px[0]:self.surface_limits_px[1]]
        save_img(rgb_cropped, 'rgb_cropped')
        depth_cropped = depth[:, self.surface_limits_px[0]:self.surface_limits_px[1]]
        rgb = cv2.resize(rgb_cropped, (386, 386), interpolation = cv2.INTER_AREA)
        depth = cv2.resize(depth_cropped, (386, 386), interpolation = cv2.INTER_AREA)
        save_img(rgb, 'rgb_cropped_386')
        save_img(depth, 'depth_cropped_386')
        print('cropped 386 depth: min:', np.min(depth), 'max:', np.max(depth))

        self.point_cloud = cv_tools.PointCloud.from_depth(depth, self.camera)
        # self.point_cloud.plot()

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Extract mask
        color_detector = cv_tools.ColorDetector()
        mask = color_detector.detect(rgb, color='yellow')
        save_img(mask, 'mask')
        target_object = clt_util.TargetObjectConvexHull(mask)
        self.target_pos_pxl = target_object.centroid.astype(np.int32)
        print('target_pos_pxl', self.target_pos_pxl)
        mask_points = np.argwhere(mask > 0)
        convex_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(convex_mask, pts=[target_object.get_limits().astype(np.int32)], color=(255, 255, 255))
        save_img(convex_mask, 'convex_mask')

        plt.imshow(convex_mask); plt.show()
        kernel = np.ones((2, 2), np.uint8)
        convex_mask = cv2.dilate(convex_mask, kernel, iterations=5)
        plt.imshow(convex_mask); plt.show()
        save_img(convex_mask, 'mask_translated_dilated')

        mask = convex_mask.copy()

        rgb_mask = rgb.copy()
        rgb_mask[convex_mask == 255] = 255
        save_img(rgb_mask, 'rgb_with_mask')

        # Extract heighmap
        plt.imshow(depth); plt.show()
        # depth[depth < 0.53] = 0
        # depth[depth > 0.65] = 0
        plt.imshow(depth); plt.show()
        max_depth = np.max(depth)
        # max_values = np.sort(depth.flatten())[::-1]
        # print(max_values[:10])
        print('max depth', max_depth)
        depth[depth == 0] = max_depth
        plt.imshow(depth); plt.show()
        depth[depth > max_depth - 0.02] = max_depth
        plt.imshow(depth); plt.show()
        heightmap = max_depth - depth
        plt.imshow(heightmap); plt.show()
        # heightmap[heightmap > 0.1] = 0
        heightmap = cv_tools.Feature(heightmap).translate(self.target_pos_pxl[0], self.target_pos_pxl[1]).crop(193, 193).array()
        save_img(heightmap, 'heightmap')
        kernel = np.ones((3, 3), np.float32) / 25
        heightmap = cv2.filter2D(heightmap, -1, kernel)
        plt.imshow(heightmap); plt.show()
        mask = cv_tools.Feature(mask).translate(self.target_pos_pxl[0], self.target_pos_pxl[1]).crop(193, 193).array()
        save_img(mask, 'mask_translated')
        self.heightmap = heightmap.copy()
        self.mask = mask.copy()
        self.target_height = self.heightmap[193, 193]
        # self.surface_distances =
        print('target height: ', self.target_height)

    def get_visual_representation(self, primitive):
        surface_distances = [0, 0, 0, 0]
        surface_distances = np.array([x / 0.5 for x in surface_distances])
        visual_feature = clt_util.get_actor_visual_feature(self.heightmap, self.mask, self.target_height/2.0,
                                                        self.finger_height, angle=0,
                                                        primitive=primitive, plot=False)
        visual_feature = torch.FloatTensor(visual_feature).reshape(1, 1, visual_feature.shape[0],
                                                                visual_feature.shape[1]).to('cpu')
        ae_output = self.autoencoder(visual_feature).detach().cpu().numpy()[0, 0, :, :]
        visual_feature = visual_feature.detach().cpu().numpy()[0, 0, :, :]
        print('max', np.max(ae_output))
        print('min', np.min(ae_output))
        ae_output[ae_output >= 0.75] = 1
        ae_output[ae_output <= 0.25 ] = 0

        # ae_output[ae_output < 1 and ae_output > 0.25] = 0.5
        save_img(visual_feature, 'visual_feature')
        save_img(ae_output, 'ae_output')
        return visual_feature, ae_output

    def get_action(self):
        # forward pass from model
        action = [0, 0, 1]
        if action[0] == 0:
            push = PushTargetDepthObjectAvoidance(angle=action[1], push_distance=action[2], push_distance_range=self.params['push']['distance'],
                                           finger_length=self.finger_height, finger_height=self.finger_height, target_height=self.target_height,
                                           camera=self.camera, pixels_to_m=self.pixels_to_m)

        push.translate(self.target_pos[:2])
        print('push init', push.get_init_pos(), 'push final', push.get_final_pos())


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
        VAE_PATH = '/home/iason/robamine_logs/2020.01.16.split_ddpg/VAE'
    elif hostname == 'iti-479':
        logging_dir = '/home/mkiatos/robamine/logs/'
    elif hostname == 'pc':
        logging_dir = '/home/ur5/iason/ral_ws/logs'
        PATH = '/home/ur5/iason/ral_ws/comms'
        VAE_PATH = '/home/ur5/iason/ral_ws/models/vae'
    else:
        raise ValueError()

    with open(yml_name, 'r') as stream:
        params = yaml.safe_load(stream)

    

    clutter = ClutterReal(params['env']['params'])
    clutter.get_heightmap()
    clutter.get_visual_representation(0)