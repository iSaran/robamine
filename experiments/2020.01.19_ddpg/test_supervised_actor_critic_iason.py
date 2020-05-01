import gym
from robamine.algo.core import SupervisedTrainWorld, EvalWorld
from robamine.algo.util import Transition
from robamine.utils.math import min_max_scale
from robamine.envs.clutter_cont import ClutterContWrapper
from robamine.envs.clutter_utils import get_action_dim, InvalidEnvError
# from robamine.algo.ddpg_torch import DDPG_TORCH
from robamine.algo.splitddpg import SplitDDPG
from robamine.algo.supervisedactorcritic import SupervisedActorCritic
from robamine import rb_logging
from robamine.utils.memory import LargeReplayBuffer, RotatedLargeReplayBuffer
import logging
import yaml
import os
import numpy as np


stream='''
agent:
  name: SupervisedActorCritic
  params:
    batch_size: 64
    device: cpu
    load_buffers: ''
    load_nets: ''
    replay_buffer_size: 1000000
    actor:
      hidden_units: [400, 300]
      learning_rate: 0.001
      preactivation_weight: 0.05
      epochs: 100
    critic:
      hidden_units: [400, 300]
      learning_rate: 0.001
      epochs: 100
    hardcoded_primitive: 2
  trainable_params: ''
env:
  name: ClutterContWrapper-v0
  params:
    hardcoded_primitive: 2
    max_timesteps: 5
    all_equal_height_prob: 0.0
    finger_size:
    - 0.005
    - 0.005
    nr_of_obstacles:
    - 1
    - 6
    render: true
    target:
      min_bounding_box: [.01, .01, .005]
      max_bounding_box: [.03, .03, .010]
      probability_box: 1.0
      enforce_convex_hull: 15
    obstacle:
      min_bounding_box: [.01, .01, .005]
      max_bounding_box: [.03, .03, .020]
      probability_box: 1.0
    push:
      distance: [0.02, 0.10]
      target_init_distance: [0.0, 0.1]
    grasp:
      spread: [0.05, 0.05]
      height: [0.01, 0.01]
      workspace: [0.0, 0.1]
    feature_normalization_per: 'session'  # available: 'session', 'episode'
    hug_probability: 0.5
world:
  comments: ''
  friendly_name: ''
  logging_dir: /home/espa/robamine_logs/2020.04.26_supervised_actor_critic/
  name: TrainEval
  params:
    episodes: 10000
    eval_episodes: 10
    eval_every: 20
    eval_render: false
    render: false
    save_every: 200
'''

# Data collection
buffer_path = '/home/espa/robamine_logs/2020.04.26_supervised_actor_critic/data/replay_buffer.hdf5'
# buffer_path = '/home/iason/Desktop/temp.hdf5'
buffer_size = 10000
buffer_episodes = 5000
rotations = 16

def train(params):
    epochs = params['agent']['params']['actor']['epochs'] + params['agent']['params']['critic']['epochs']
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name=params['world']['friendly_name'], file_level=logging.INFO)
    trainer = SupervisedTrainWorld(agent=params['agent'], dataset=buffer_path, epochs=epochs, save_every=20)
    trainer.run()

def compile_dataset(params):
    env = ClutterContWrapper(params)
    buffer = RotatedLargeReplayBuffer(buffer_size, 386, env.action_dim[0] + 1, buffer_path, rotations=rotations)

    for i in range(buffer_episodes):
        transition = run_episode(env, i)
        buffer.store(transition)

def run_episode(env, i):
    print('Episode ', i, ' from ', buffer_episodes)

    try:
        state = env.reset()
        state_dict = env.state_dict()

        action_to_store = np.zeros((rotations, get_action_dim(env.params['hardcoded_primitive'])[0] + 1))
        reward_to_store = np.zeros((rotations, 1))
        for phi in range(rotations):
            step = 180 / rotations
            theta = phi * step
            theta = np.random.normal(theta, step/4)
            theta = min(theta, 180)
            theta = max(theta, 0)
            theta = min_max_scale(theta, range=[0, 180], target_range=[-1, 1])
            action = np.array([0, theta])
            next_state, reward, done, info = env.step(action)

            action_to_store[phi, :] = action
            reward_to_store[phi, :] = reward
            env.load_state_dict(state_dict)
        transition = Transition(state, action_to_store, reward_to_store, None, None)
    except InvalidEnvError as e:
        print("WARN: {0}. Invalid environment episode in data collection. A new environment will be spawn.".format(e))
        return run_episode(env, i)
    return transition


def eval(params):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name=params['world']['friendly_name'], file_level=logging.INFO)
    agent = SupervisedActorCritic.load('/home/espa/robamine_logs/2020.04.26_supervised_actor_critic/robamine_logs_2020.04.27.12.23.39.143687/model.pkl')
    params['env']['params']['render'] = True
    eval = EvalWorld(agent=agent, env=params['env'], params={'episodes': 5, 'render': True})
    eval.run()

def eval_in_scenes(model_path='/home/espa/robamine_logs/2020.04.26_supervised_actor_critic/robamine_logs_2020.04.30.16.41.17.945353/model.pkl'):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name=params['world']['friendly_name'], file_level=logging.INFO)
    agent = SupervisedActorCritic.load(model_path)
    params['env']['params']['render'] = False
    params['env']['params']['safe'] = False
    # params['env']['params']['seed'] = 5
    saved_scenes = np.arange(0, 1000, 1).tolist()
    eval = EvalWorld(agent=agent, env=params['env'], params={'episodes': len(saved_scenes), 'render': False})
    eval.seed_list = saved_scenes
    eval.run()



if __name__ == '__main__':
    params = yaml.safe_load(stream)
    # compile_dataset(params['env']['params'])
    # train(params)
    # eval(params)
    eval_in_scenes()
