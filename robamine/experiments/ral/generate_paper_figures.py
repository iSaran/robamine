from robamine.algo.core import EvalWorld, TrainEvalWorld
from robamine import rb_logging
import logging
import yaml
import socket
import numpy as np
import os
import gym
import copy
import robamine.algo.core as core
import robamine.algo.util as algo_util
import torch
import torch.nn as nn
import robamine.algo.conv_vae as ae
import torch.optim as optim
import robamine.utils.memory as rb_mem
import robamine.envs.clutter_utils as clutter
import pickle
from robamine.envs.clutter_cont import ClutterContWrapper
import math
import matplotlib.pyplot as plt
from robamine.experiments.ral.supervised_push_obstacle import PushObstacleFeature

logger = logging.getLogger('robamine')


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
    vae_path = os.path.join(logging_dir, 'VAE')
    params['env']['params']['vae_path'] = vae_path

    params['env']['params']['render'] = True
    env = ClutterContWrapper(params=params['env']['params'])

    obs = env.reset(seed=2)

    # Load autoencoder and scaler
    ae_path = os.path.join(vae_path, 'model.pkl')
    with open(ae_path, 'rb') as file1:
        model = torch.load(file1, map_location='cpu')
    latent_dim = model['encoder.fc.weight'].shape[0]
    ae_params = ae.params
    ae_params['device'] = 'cpu'
    autoencoder = ae.ConvVae(latent_dim, ae_params)
    autoencoder.load_state_dict(model)

    fig, ax = plt.subplots()

    img = env.env.heightmap_raw
    name = 'heightmap'
    path = os.path.join(os.path.join(logging_dir, 'generate_paper_figures'), name)
    plt.imsave(path, img, cmap='gray', vmin=np.min(img), vmax=np.max(img))

    img = env.env.mask_raw
    name = 'mask'
    path = os.path.join(os.path.join(logging_dir, 'generate_paper_figures'), name)
    plt.imsave(path, img, cmap='gray', vmin=np.min(img), vmax=np.max(img))

    img = env.env.mask_raw
    path = os.path.join(os.path.join(logging_dir, 'generate_paper_figures'), name)
    plt.imsave(path, img, cmap='gray', vmin=np.min(img), vmax=np.max(img))

    name = 'haha'
    heightmap = obs['heightmap_mask'][0].copy()
    mask = obs['heightmap_mask'][1].copy()
    target_bounding_box_z = obs['target_bounding_box'][2].copy()
    finger_height = obs['finger_height'].copy()
    surface_distances = [obs['surface_size'][0] - obs['object_poses'][0, 0], \
                              obs['surface_size'][0] + obs['object_poses'][0, 0], \
                              obs['surface_size'][1] - obs['object_poses'][0, 1], \
                              obs['surface_size'][1] + obs['object_poses'][0, 1]]
    surface_distances = np.array([x / 0.5 for x in surface_distances])
    target_pos = obs['object_poses'][0, 0:2].copy()
    visual_feature = clutter.get_actor_visual_feature(heightmap, mask, target_bounding_box_z,
                                                      finger_height, 0,
                                                      0, plot=False)
    visual_feature = torch.FloatTensor(visual_feature).reshape(1, 1, visual_feature.shape[0],
                                                               visual_feature.shape[1]).to('cpu')
    ae_output = autoencoder(visual_feature).detach().cpu().numpy()[0, 0, :, :]
    visual_feature = visual_feature.detach().cpu().numpy()[0, 0, :, :]

    img = visual_feature
    name = 'feature'; path = os.path.join(os.path.join(logging_dir, 'generate_paper_figures'), name)
    plt.imsave(path, img, cmap='gray', vmin=np.min(img), vmax=np.max(img))

    img = ae_output
    name = 'ae_output'; path = os.path.join(os.path.join(logging_dir, 'generate_paper_figures'), name)
    plt.imsave(path, img, cmap='gray', vmin=np.min(img), vmax=np.max(img))

