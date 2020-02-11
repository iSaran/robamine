import gym
import yaml
import robamine
import numpy as np
from robamine.algo.core import TrainingEpisode
from robamine.algo.splitddpg import SplitDDPG

yaml_file = '''
agent:
  name: SplitDDPG
  params:
    actions: [3]
    batch_size: [64, 64]
    device: cpu
    gamma: 0.99
    load_buffers: ''
    load_nets: ''
    replay_buffer_size: 1000000
    tau: 0.001
    update_iter: [1, 1]
    actor:
      hidden_units: [[400, 300], [400, 300]]
      learning_rate: 0.001
    critic:
      hidden_units: [[400, 300], [400, 300]]
      learning_rate: 0.001
    noise:
      name: Normal
      sigma: 0.2
    epsilon:
      start: 0.9
      end: 0.05
      decay: 10000
    heightmap_rotations: 4
  trainable_params: ''
env:
  name: ClutterCont-v0
  params:
    all_equal_height_prob: 0.0
    finger_size:
    - 0.005
    - 0.005
    nr_of_obstacles:
    - 1
    - 8
    obstacle_height_range:
    - 0.005
    - 0.02
    obstacle_probability_box: 1.0
    render: render
    target_height_range:
    - 0.005
    - 0.01
    target_probability_box: 1.0
    push:
      distance: [0.05, 0.10]
      target_init_distance: [0.0, 0.1]
      minimum_distance: rectangle
    grasp:
      spread: [0.05, 0.05]
      height: [0.01, 0.01]
      workspace: [0.0, 0.1]
    heightmap_rotations: 4
    maximum_timesteps: 1
world:
  comments: ''
  friendly_name: ''
  logging_dir: /home/iason/Dropbox/projects/phd/clutter/training/2020.01.16.split_ddpg
  name: TrainEval
  params:
    episodes: 5000
    eval_episodes: 10
    eval_every: 50
    eval_render: false
    render: false
    save_every: 200
'''

def run():
    params = yaml.safe_load(yaml_file)

    env = gym.make(params['env']['name'], params=params['env']['params'])
    env.seed(0)

    # Grasp
    theta = 0.4
    push_distance = 1  # distance from target
    distance = 0
    action = np.array([0, theta, push_distance, distance])
    env.reset()
    obs, reward, done, info = env.step(action)
    print('shape of obs:', obs.shape)

    env.seed(0)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    print('state dim', state_dim)
    print('action dim', action_dim)
    agent = SplitDDPG(state_dim = state_dim, action_dim = action_dim, params=params['agent']['params'])
    episode = TrainingEpisode(agent, env)
    episode.run()

if __name__ == '__main__':
    run()
