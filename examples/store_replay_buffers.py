import argparse
import yaml
import gym
from robamine.utils.memory import ReplayBuffer
from robamine.algo.util import Transition
import numpy as np
import robamine as rm
import logging
import os

def run(yml):
    with open("../yaml/" + yml + ".yml", 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            rm.rb_logging.init(directory=params['logging_directory'], file_level=logging.INFO)
            logger = logging.getLogger('robamine')
            replay_buffer = []
            for i in range(params['nr_primitives']):
                replay_buffer.append(ReplayBuffer(params['buffer_size']))
            env = gym.make(params['env']['name'], params=params['env'])
            timestep = 0
            while True:
                if timestep >= params['timesteps']:
                    break
                observation = env.reset()

                while True:
                    if timestep >= params['timesteps']:
                        break
                    timestep += 1
                    action = env.action_space.sample()
                    observation_new, reward, done, info = env.step(action)
                    transition = Transition(observation, action, reward, observation_new, done)
                    replay_buffer[int(np.floor(transition.action / params['nr_substates']))].store(transition)
                    observation = observation_new.copy()
                    if done:
                        break

                print('Timestep: ', timestep, 'Buffer sizes:', replay_buffer[0].size(), replay_buffer[1].size(), replay_buffer[2].size())

                log_dir = rm.rb_logging.get_logger_path()
                for i in range(params['nr_primitives']):
                    replay_buffer[i].save(os.path.join(log_dir, 'replay_buffer' + str(i) + '.pkl'))

        except yaml.YAMLError as exc:
            print(exc)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--yml', type=str, default='store_buffers', help='The yaml file to load')
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

if __name__ == '__main__':
    args = parse_args()
    run(**args)
