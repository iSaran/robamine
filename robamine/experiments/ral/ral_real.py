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

import robamine.utils.cv_tools as cv_tools
import robamine.envs.clutter_utils as clt_util
import robamine.algo.conv_vae as ae

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

class ClutterReal:
    def __init__(self):
        # self.camera = cv_tools.PinholeCamera(fovy, self.size)
        self.camera = None
        self.heightmap = None
        self.mask = None
        self.centroid_pxl = None

        # Surface limits in pixels
        self.surface_limits = [0.25, -0.25, 0.25, -0.25]
        self.surface_limits_px = [330, 1045, 0, 720]
        self.finger_height = 0.001
        self.surface_centroid_px = [self.surface_limits_px[1] - self.surface_limits_px[0] / 2, 
                                    self.surface_limits_px[3] - self.surface_limits_px[2] / 2, ]


        # Load autoencoder and scaler
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


        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Extract mask
        color_detector = cv_tools.ColorDetector()
        mask = color_detector.detect(rgb, color='yellow')
        save_img(mask, 'mask')
        target_object = clt_util.TargetObjectConvexHull(mask)
        self.centroid_pxl = target_object.centroid.astype(np.int32)
        print('centroid_pxl',self.centroid_pxl)
        mask_points = np.argwhere(mask > 0)
        convex_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(convex_mask, pts=[target_object.get_limits().astype(np.int32)], color=(255, 255, 255))
        mask = convex_mask.copy()
        save_img(mask, 'convex_mask')

        # Extract heighmap
        depth[depth > 0.7] = 0
        max_depth = np.max(depth)
        # max_values = np.sort(depth.flatten())[::-1]
        # print(max_values[:10])
        print('max depth', max_depth)
        depth[depth == 0] = max_depth
        depth[depth > max_depth - 0.02] = max_depth
        heightmap = max_depth - depth
        heightmap = cv_tools.Feature(heightmap).translate(self.centroid_pxl[0], self.centroid_pxl[1]).crop(193, 193).array()
        save_img(heightmap, 'heightmap')
        mask = cv_tools.Feature(mask).translate(self.centroid_pxl[0], self.centroid_pxl[1]).crop(193, 193).array()
        save_img(mask, 'mask_translated')
        self.heightmap = heightmap.copy()
        self.mask = mask.copy()
        self.target_height = self.heightmap[193, 193]
        self.surface_distances = 
        print('target height: ', self.target_height)

    
    def get_visual_representation(obs, primitive):
        surface_distances = [obs['surface_size'][0] - obs['object_poses'][0, 0], \
                            obs['surface_size'][0] + obs['object_poses'][0, 0], \
                            obs['surface_size'][1] - obs['object_poses'][0, 1], \
                            obs['surface_size'][1] + obs['object_poses'][0, 1]]
        surface_distances = np.array([x / 0.5 for x in surface_distances])
        target_pos = obs['object_poses'][0, 0:2].copy()
        visual_feature = clutter.get_actor_visual_feature(self.heightmap, self.mask, self.target_height/2.0,
                                                        self.finger_height, angle=0,
                                                        primitive=primitive, plot=False)
        visual_feature = torch.FloatTensor(visual_feature).reshape(1, 1, visual_feature.shape[0],
                                                                visual_feature.shape[1]).to('cpu')
        ae_output = autoencoder(visual_feature).detach().cpu().numpy()[0, 0, :, :]
        visual_feature = visual_feature.detach().cpu().numpy()[0, 0, :, :]
        save_img(visual_feature, 'visual_feature')
        save_img(ae_output, 'ae_output')
        return visual_feature, ae_output



if __name__ == '__main__':
    pid = os.getpid()
    print('Process ID:', pid)
    hostname = socket.gethostname()

    yml_name = 'params.yml'
    if hostname == 'dream':
        logging_dir = '/home/espa/robamine_logs/'
    elif hostname == 'triss':
        logging_dir = '/home/iason/robamine_logs/2020.01.16.split_ddpg/'
        PATH = '/home/iason/ral_exp'
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

    

    clutter = ClutterReal()
    clutter.get_heightmap()