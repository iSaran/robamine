import gym
import yaml
import robamine
import numpy as np

env_yaml = '''
name: ClutterCont-v0
params:
  all_equal_height_prob: 0.0
  finger_size:
  - 0.005
  - 0.005
  nr_of_obstacles:
  - 7
  - 7
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
'''

def run():
    params = yaml.safe_load(env_yaml)

    env = gym.make(params['name'], params=params['params'])

    # Grasp
    env.seed(0)
    angle_wrt_target = 0.4
    distance_from_target = -1  # distance from target
    angle_second_finger_wrt_first = 0.85
    action = np.array([3, angle_wrt_target, distance_from_target, angle_second_finger_wrt_first])
    env.reset()
    env.step(action)

    # Empty Grasp
    angle_wrt_target = 0.4
    distance_from_target = -1  # distance from target
    angle_second_finger_wrt_first = 0.85
    action = np.array([3, angle_wrt_target, distance_from_target, angle_second_finger_wrt_first])
    env.step(action)

    # Collision grasp
    angle_wrt_target = 0.4
    distance_from_target = -1  # distance from target
    angle_second_finger_wrt_first = 0.6
    action = np.array([3, angle_wrt_target, distance_from_target, angle_second_finger_wrt_first])
    env.step(action)

    # Failed grasp
    env.seed(0)
    angle_wrt_target = 0.4
    distance_from_target = -1  # distance from target
    angle_second_finger_wrt_first = 0.7
    action = np.array([3, angle_wrt_target, distance_from_target, angle_second_finger_wrt_first])
    env.reset()
    env.step(action)

if __name__ == '__main__':
    run()
