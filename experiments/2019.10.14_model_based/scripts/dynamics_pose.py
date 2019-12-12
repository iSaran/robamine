from robamine.envs.clutter import Clutter, get_2d_displacement
from robamine.algo.splitdynamicsmodelpose import SplitDynamicsModelPose

import pickle
import gym

if __name__ == '__main__':
    # Create env
    params = {
      'discrete': True,
      'nr_of_actions': 16,  # u x w
      'render': True,
      'nr_of_obstacles': [0, 0],
      'target_probability_box': 1.0,
      'target_height_range': [0.01, 0.01],
      'obstacle_probability_box': 1.0,
      'obstacle_height_range': [0.005, 0.005],
      'push_distance': [0.25, 0.25],
      'split': False,
      'extra_primitive': False,
      'all_equal_height_prob': 0.0,
      'finger_size': [0.005, 0.005]
    }
    env = gym.make('Clutter-v0', params=params)
    # env.seed(23)
    state = env.reset()
    action = 2
    next_state, reward, done, info = env.step(action)
    print('displacement:', info['extra_data']['displacement'][2])
    print('handcrafted predicted_displacement:', info['extra_data']['predicted_displacement'])
    force_vel_data = [info['extra_data']['push_finger_forces'], info['extra_data']['push_finger_vel']]


    # model = SplitDynamicsModelPoseLSTM.load('/home/iason/Dropbox/projects/phd/clutter/training/2019.10.14_model_based/dynamics_model_pose_2/model.pkl')
    # prediction = model.predict(force_vel_data, action)
    # print('lstm predicted_displacement:', prediction)

    model = SplitDynamicsModelPose.load('/home/iason/Dropbox/projects/phd/clutter/training/2019.10.14_model_based/dynamics_model_pose_fc_position/model.pkl')
    prediction = model.predict(force_vel_data, action)
    print('lstm predicted_displacement with no filtering:', prediction)
