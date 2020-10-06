from robamine.algo.core import EvalWorld, TrainEvalWorld
from robamine.algo.splitddpg import SplitDDPG, Critic, Actor, ObstacleAvoidanceLoss
from robamine import rb_logging
import logging
import yaml
import socket
import numpy as np
import os
import pickle
import cv2

import torch
import torch.nn as nn
import torch.optim as optim

import robamine.envs.clutter_utils as clutter
import h5py
from robamine.utils.memory import ReplayBuffer
from robamine.algo.util import OrnsteinUhlenbeckActionNoise, NormalNoise, Transition

import robamine.utils.cv_tools as cv_tools
import robamine.envs.clutter_utils as clt_util
import matplotlib.pyplot as plt

logger = logging.getLogger('robamine')

PATH = '/home/iason/ral_exp'

class ClutterReal:
    def __init__(self):
        # self.camera = cv_tools.PinholeCamera(fovy, self.size)
        self.camera = None

    def get_heightmap(self):
        # Load grabbed images
        with open(os.path.join(PATH, 'rgbd.pkl'), 'rb') as f:
            rgb, depth = pickle.load(f, encoding='latin1')

        color_detector = cv_tools.ColorDetector()
        mask = color_detector.detect(rgb, color='yellow')

        target_object = clt_util.TargetObjectConvexHull(mask)
        centroid_pxl = target_object.centroid.astype(np.int32)

        # Create a convex mask
        mask_points = np.argwhere(mask > 0)
        convex_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(convex_mask, pts=[target_object.get_limits().astype(np.int32)], color=(255, 255, 255))
        mask = convex_mask

        cv2.imshow('depth', depth)
        cv2.waitKey()
        plt.imshow(mask)
        plt.show()

        self.heightmap_raw = cv_tools.Feature(heightmap).crop(193, 193).array()
        self.mask_raw = cv_tools.Feature(mask).crop(193, 193).array()

        # # Calculate the centroid w.r.t. initial image (640x480) in pixels
        #
        #
        # # Calculate the centroid and the target pos w.r.t. world
        # z = depth[centroid_pxl[1], centroid_pxl[0]]
        # centroid_image = camera.back_project(centroid_pxl, z)
        # centroid_camera = np.matmul(self.rgb_to_camera_frame, centroid_image)
        # camera_pose = get_camera_pose(self.sim, 'xtion')  # g_wc: camera w.r.t. the world
        # self.target_pos_vision = np.matmul(camera_pose,
        #                                    np.array([centroid_camera[0], centroid_camera[1], centroid_camera[2], 1.0]))[:3]
        # self.target_pos_vision[2] /= 2.0
        #
        # self.heightmap_raw = cv_tools.Feature(heightmap).crop(193, 193).array()
        # self.mask_raw = cv_tools.Feature(mask).crop(193, 193).array()

if __name__ == '__main__':
    pid = os.getpid()
    print('Process ID:', pid)
    hostname = socket.gethostname()

    yml_name = 'params.yml'
    if hostname == 'dream':
        logging_dir = '/home/espa/robamine_logs/'
    elif hostname == 'triss':
        logging_dir = '/home/iason/robamine_logs/2020.01.16.split_ddpg/'
    elif hostname == 'iti-479':
        logging_dir = '/home/mkiatos/robamine/logs/'
    else:
        raise ValueError()

    with open(yml_name, 'r') as stream:
        params = yaml.safe_load(stream)

    # logging_dir = '/tmp'
    params['world']['logging_dir'] = logging_dir
    params['env']['params']['vae_path'] = os.path.join(logging_dir, 'VAE')
    params['agent']['params']['actor']['autoencoder']['model'] = os.path.join(os.path.join(logging_dir, 'VAE'), 'model.pkl')
    params['agent']['params']['actor']['autoencoder']['scaler'] = os.path.join(os.path.join(logging_dir, 'VAE'), 'normalizer.pkl')
    params['env']['params']['push']['obstacle_avoid'] = False

    clutter = ClutterReal()
    clutter.get_heightmap()