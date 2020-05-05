import gym
from robamine.algo.core import SupervisedTrainWorld, EvalWorld, InvalidEnvError
from robamine.algo.util import Transition
from robamine.utils.math import min_max_scale
from robamine.envs.clutter_cont import ClutterContWrapper, ClutterCont
from robamine.envs.clutter_utils import get_action_dim, get_observation_dim, obs_dict2feature
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
    shapes = ClutterCont.get_obs_shapes()
    buffer = RotatedLargeReplayBuffer(buffer_size, shapes, env.action_dim[0] + 1, buffer_path, rotations=rotations)

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


def merge_buffers(path='/home/espa/robamine_logs/2020.04.26_supervised_actor_critic/data/', merged_name='buffer_all',
                  buffer_names=['1', '2']):
    buffers = []
    for name in buffer_names:
        buffers.append(RotatedLargeReplayBuffer.load(os.path.join(path, name + '.hdf5')))

    shapes = ClutterCont.get_obs_shapes()
    action_dim = buffers[0].file['action'].shape[2]
    rotations = buffers[0].file['action'].shape[1]
    buffer = RotatedLargeReplayBuffer(buffer_size, shapes, action_dim, os.path.join(path, merged_name + '.hdf5'),
                                      rotations=rotations)

    for b in buffers:
        buffer.merge(b)


def preprocess_dataset(path='/home/espa/robamine_logs/2020.04.26_supervised_actor_critic/data/', name='buffer',
                       target_name='feature_buffer', primitive=1):
    '''Transforms a heightmap dataset to a feature dataset'''
    old = RotatedLargeReplayBuffer.load(os.path.join(path, name + '.hdf5'))
    shape = {'feature': (get_observation_dim(primitive),)}
    new = RotatedLargeReplayBuffer(old.buffer_size, shape, old.action_dim, os.path.join(path, target_name + '.hdf5'),
                                   existing=False, rotations=old.rotations)

    for i in range(old.n_scenes):
        scene = old.get_scene(i)
        feature = obs_dict2feature(primitive=primitive, obs_dict=scene.state, angle=0).array()
        transition = Transition({'feature': feature}, scene.action, scene.reward, None, None)
        new.store(transition)

def run_one_transition_in_clutter(params):
    '''Just to test quickly changes in clutter'''
    params['render'] = True
    env = ClutterContWrapper(params)
    state = env.reset()
    action = np.array([0, 0, 0, 0])
    env.step(action)


if __name__ == '__main__':
    params = yaml.safe_load(stream)
    # compile_dataset(params['env']['params'])
    # merge_buffers()
    # train(params)
    # eval(params)
    # eval_in_scenes()
    run_one_transition_in_clutter(params['env']['params'])
