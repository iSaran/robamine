from robamine.algo.core import SampleTransitionsWorld, EvalWorld
from robamine.algo.splitddpg import SplitDDPG
from robamine.algo.util import EnvData, Transition
from robamine.utils.memory import ReplayBuffer
from robamine import rb_logging
import logging
import yaml
import os
import numpy as np

from robamine.envs.clutter_utils import rotated_to_regular_transitions
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

stream='''
agent:
  name: RandomHybrid
  params:
      actions: [1]
      noise:
        name: Normal
        sigma: 0.2
env:
  name: ClutterContWrapper-v0
  params:
    hardcoded_primitive: 2
    all_equal_height_prob: 0.0
    finger_size:
    - 0.005
    - 0.005
    nr_of_obstacles:
    - 1
    - 3
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
    heightmap_rotations: 4
    max_timesteps: 5
world:
  name: 'SampleTransitions'
  logging_dir: /tmp/robamine_logs/
  friendly_name: ''
  comments: ''
  params:
    transitions: 20
    render: false
world_eval:
  name: 'EvalWorld'
  logging_dir: /tmp/robamine_logs/
  friendly_name: ''
  comments: ''
  params:
    episodes: 20
    render: false
'''

ddpg_stream='''
agent:
  name: SplitDDPG
  params:
    actions: [1]
    batch_size: [128, 128]
    device: cpu
    gamma: 0.99
    load_buffers: ''
    load_nets: ''
    replay_buffer_size: 1000000
    tau: 0.001
    update_iter: [1, 1]
    actor:
      hidden_units: [[400, 300], [400, 300]]
      learning_rate: 0.0001
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
    # load_actors:  # Uncomment to use pretrained actors. Comment-out for initialize actors from scratch it if want to initialize actors from scratch
    #   - /home/iason/Dropbox/projects/phd/clutter/training/2020.01.16.split_ddpg/robamine_logs_2020.02.05.15.20.58.472410_bck/model.pkl
    #   - /home/iason/Dropbox/projects/phd/clutter/training/2020.01.16.split_ddpg/robamine_logs_2020.02.05.15.20.58.472410_bck/model.pkl
  trainable_params: ''
'''

def run():
    # compile_data()
    path = '/home/iason/Dropbox/projects/phd/clutter/training/2020.03.10_grasp_target/recorded_data/robamine_logs_2020.03.10.19.17.54.29752'
    # samples = EnvData.load(os.path.join(path, 'samples.env'))
    # params = yaml.safe_load(stream)
    # trans = rotated_to_regular_transitions(samples.transitions, params['env']['params']['heightmap_rotations'])
    # buffer = ReplayBuffer(10000000)
    # for i in trans:
    #     buffer.store(i)
    # buffer.save(os.path.join(path, 'buffer'))

    train_ddpg_supervised(path)

    eval_ddpg(path)

def compile_data():
    params = yaml.safe_load(stream)
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name=params['world']['friendly_name'], file_level=logging.INFO)
    world = SampleTransitionsWorld(agent=params['agent'], env=params['env'], params=params['world']['params'], name='SampleTransitionsWorld')
    world.run()

    samples = EnvData.load(os.path.join(world.log_dir, 'samples.env'))
    trans = rotated_to_regular_transitions(samples.transitions, params['env']['params']['heightmap_rotations'])
    buffer = ReplayBuffer(10000000)
    for i in trans:
        buffer.store(i)

    buffer.save(os.path.join(world.log_dir, 'buffer'))
    print('Saved buffer in ', world.log_dir)

def train_ddpg_supervised(path):
    # loaded_buffer = ReplayBuffer.load(os.path.join(path, 'buffer'))

    samples = EnvData.load(os.path.join(path, 'samples.env'))
    params = yaml.safe_load(stream)
    trans = rotated_to_regular_transitions(samples.transitions, params['env']['params']['heightmap_rotations'])
    buffer = ReplayBuffer(10000000)
    buffer_low = ReplayBuffer(10000000)
    buffer_high = ReplayBuffer(10000000)
    for i in trans:
        if i.reward == -1:
            buffer_low.store(i)
        else:
            buffer_high.store(i)

        buffer.store(i)

    print('Buffer low size:', buffer_low.size())
    print('Buffer high size:', buffer_high.size())

    ddpg_params = yaml.safe_load(ddpg_stream)['agent']['params']
    print('hahah', len(buffer.buffer[0].state))
    ddpg = SplitDDPG(state_dim=len(buffer.buffer[0].state) * 4, action_dim=0, params=ddpg_params)
    ddpg.replay_buffer[0] = buffer

    actor_loss = []
    tanh_out_mean = []
    tanh_out_max = []
    tanh_out_min = []
    q_value_mean = []
    steps = 1300
    for i in range(steps):
        print('-------- iteration: ', i)

        # Calculate batch for training and learn
        batch_high = buffer_high.sample_batch(int(ddpg_params['batch_size'][0] / 2))
        batch_low = buffer_low.sample_batch(int(ddpg_params['batch_size'][0] / 2))
        batch = Transition(np.concatenate((batch_high.state, batch_low.state)),
                           np.concatenate((batch_high.action, batch_low.action)),
                           np.concatenate((batch_high.reward, batch_low.reward)),
                           np.concatenate((batch_high.next_state, batch_low.next_state)),
                           np.concatenate((batch_high.terminal, batch_low.terminal)))

        # batch = buffer.sample_batch(ddpg_params['batch_size'][0])
        # print(batch.reward)

        ddpg.update_net(0, batch)
        # print('  actor loss:', ddpg.info['actor_0_loss'])
        # print('  critic loss:', ddpg.info['critic_0_loss'])

        # Calculate batch from total training for logging data
        batch = buffer.sample_batch(buffer.size())
        state = torch.FloatTensor(batch.state).to(ddpg.device)
        _, action_ = ddpg.get_low_level_action(batch.action)
        action = torch.FloatTensor(action_).to(ddpg.device)
        output = ddpg.actor[0].forward2(state)
        tanh_out_mean.append(np.asscalar(torch.mean(output).cpu().detach().numpy()))
        tanh_out_max.append(np.asscalar(torch.max(output).cpu().detach().numpy()))
        tanh_out_min.append(np.asscalar(torch.min(output).cpu().detach().numpy()))

        value = ddpg.critic[0](state, action)
        q_value_mean.append(np.asscalar(torch.mean(value).cpu().detach().numpy()))
        # print('     max: ', torch.max(output))
        # print('     min: ', torch.min(output))

        l = np.asscalar(nn.functional.mse_loss(ddpg.actor[0](state), action).cpu().detach().numpy())
        actor_loss.append(l)
        # print(' :', ddpg.actor.forward2(ddpg.replay_buffer[0]))
        # print('weight', ddpg.actor[0].out.weight)
        # print('bias', ddpg.actor[0].out.bias)

        plots = [{'signal': actor_loss, 'title': 'Actor Loss', 'legend': None},
                 {'signal': tanh_out_mean, 'title': 'Mean without tanh', 'legend': None},
                 {'signal': tanh_out_min, 'title': 'Min without tanh', 'legend': None},
                 {'signal': tanh_out_max, 'title': 'Max without tanh', 'legend': None},
                 {'signal': q_value_mean, 'title': 'Mean q value', 'legend': None},
                 ]

    ddpg.save(os.path.join(path, 'model.pkl'))
    plot(plots, steps)

def eval_ddpg(path):
    ddpg = SplitDDPG.load(os.path.join(path, 'model.pkl'))

    params = yaml.safe_load(stream)
    params['env']['params']['render'] = True
    rb_logging.init(directory=params['world_eval']['logging_dir'], friendly_name=params['world_eval']['friendly_name'], file_level=logging.INFO)
    world = EvalWorld(agent=ddpg, env=params['env'], params=params['world_eval']['params'])
    world.run()


def plot(plots, steps, smoothing=0.9):
    def smooth(scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return smoothed

    fig, axs = plt.subplots(3, 3, sharex=False, sharey=False)
    axs = axs.ravel()
    x = np.arange(0, steps, 1)

    # add key: smoothing if you want different smoothing for a plot

    for i in range(len(plots)):
        if plots[i]['legend']:
            for j in range(len(plots[i]['legend'])):
                axs[i].plot(x, smooth(plots[i]['signal'][j], plots[i].get('smoothing', smoothing)))
            axs[i].legend(plots[i]['legend'])
        else:
            axs[i].plot(x, smooth(plots[i]['signal'], plots[i].get('smoothing', smoothing)))

        axs[i].set_title(plots[i]['title'])
    plt.show()

if __name__ == '__main__':
    run()
