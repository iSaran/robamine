import argparse
import yaml
import gym
from robamine.utils.memory import ReplayBuffer
from robamine.algo.core import Transition
import numpy as np
import robamine as rm
import logging
import os

if __name__ == '__main__':
    path = '/home/iason/Dropbox/projects/phd/clutter/clutter-shared/training/robamine_logs_2019.09.04.15.00.34.819989_replay_buffers'

    # Load and merge buffers
    replay_buffer = ReplayBuffer.load(os.path.join(path, 'replay_buffer' + str(0) + '.pkl'))
    replay = ReplayBuffer.load(os.path.join(path, 'replay_buffer' + str(1) + '.pkl'))
    replay_buffer.merge(replay)
    replay = ReplayBuffer.load(os.path.join(path, 'replay_buffer' + str(2) + '.pkl'))
    replay_buffer.merge(replay)

    # Keep only the first feature from state and next state
    for i in range(replay_buffer.size()):
        state = np.split(replay_buffer(i).state, 8)
        replay_buffer(i).state = state[0].copy()
        next_state = np.split(replay_buffer(i).next_state, 8)
        replay_buffer(i).next_state = next_state[0].copy()

    replay_buffer.save(os.path.join(path, 'buffer_dqn_extra_primitive.pkl'))

