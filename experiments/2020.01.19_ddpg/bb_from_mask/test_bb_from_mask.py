import gym
import yaml
import robamine
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import acos, pi
import imageio
from robamine.utils.cv_tools import ColorDetector
from robamine.utils.orientation import rot_x, rot2angleaxis
from math import pi


env_yaml = '''
name: ClutterContToTestBBFromMask-v0
params:
  all_equal_height_prob: 0.0
  finger_size:
  - 0.005
  - 0.005
  nr_of_obstacles:
  - 0
  - 0
  obstacle_height_range:
  - 0.005
  - 0.02
  obstacle_probability_box: 1.0
  render: true
  target_height_range:
  - 0.005
  - 0.01
  target_probability_box: 1.0
  max_timesteps: 5
  push:
    distance: [0.05, 0.10]
    target_init_distance: [0.0, 0.1]
    minimum_distance: rectangle
  grasp:
    spread: [0.05, 0.05]
    height: [0.01, 0.01]
    workspace: [0, 0.1]
  heightmap_rotations: 16
'''

def run():
    params = yaml.safe_load(env_yaml)
    env = gym.make(params['name'], params=params['params'])
    cd = ColorDetector('red')

    # Grasp
    env.seed(0)
    obss = env.reset()
    fig, ax = plt.subplots(4, 4)
    ax = ax.ravel()
    plt_i = 0
    for obs in obss:
        if plt_i < 13:
            obs[obs > 0] = 1
            # Create figure and axes
            rot_mat, bb = cd.get_bounding_box(obs, plot=True)
            plt_i += 1

    plt_i = 13
    custom_mask = imageio.imread('custom_mask_3.png')
    custom_mask = custom_mask[:, :, 0]
    rot_mat, bb = cd.get_bounding_box(custom_mask, plot=True)

    plt_i = 14
    custom_mask = imageio.imread('custom_mask.png')
    custom_mask = custom_mask[:, :, 0]
    rot_mat, bb = cd.get_bounding_box(custom_mask, plot=True)

    plt_i = 15
    custom_mask = imageio.imread('custom_mask_2.png')
    custom_mask = custom_mask[:, :, 0]
    rot_mat, bb = cd.get_bounding_box(custom_mask)

    plt.show()

if __name__ == '__main__':
    run()
