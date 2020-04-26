import gym
from robamine.algo.core import SupervisedTrainWorld, EvalWorld
from robamine.algo.util import Transition
from robamine.envs.clutter_cont import ClutterContWrapper
# from robamine.algo.ddpg_torch import DDPG_TORCH
from robamine.algo.splitddpg import SplitDDPG
from robamine.algo.supervisedactorcritic import SupervisedActorCritic
from robamine import rb_logging
from robamine.utils.memory import LargeReplayBuffer
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
    render: false
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
buffer_size = 10000
buffer_episodes = 1000

def train(params):
    epochs = params['agent']['params']['actor']['epochs'] + params['agent']['params']['critic']['epochs']
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name=params['world']['friendly_name'], file_level=logging.INFO)
    trainer = SupervisedTrainWorld(agent=params['agent'], dataset=buffer_path, epochs=epochs, save_every=20)
    trainer.run()

def compile_dataset(params):
    env = ClutterContWrapper(params)
    buffer = LargeReplayBuffer(buffer_size, 386, env.action_dim[0] + 1, buffer_path)

    for i in range(buffer_episodes):

        state = env.reset()
        while True:
            action = np.array([0, np.random.uniform(-1, 1)])
            next_state, reward, done, info = env.step(action)
            transition = Transition(state, action, reward, next_state, done)
            state = next_state.copy()
            if done:
                break
            buffer.store(transition)
        print('Episode ', i, ' from ', buffer_episodes)

def eval(params):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name=params['world']['friendly_name'], file_level=logging.INFO)
    agent = SupervisedActorCritic.load('/home/espa/robamine_logs/2020.04.26_supervised_actor_critic/robamine_logs_2020.04.27.12.23.39.143687/model.pkl')
    params['env']['params']['render'] = True
    eval = EvalWorld(agent=agent, env=params['env'], params={'episodes': 5, 'render': True})
    eval.run()

if __name__ == '__main__':
    params = yaml.safe_load(stream)
    # compile_dataset(params['env']['params'])
    # train(params)
    eval(params)
