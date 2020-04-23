import gym
import yaml
import numpy as np
import time
from robamine.envs.clutter_utils import get_rotated_transition, obs_dict2feature
from robamine.utils.cv_tools import Feature
from robamine.utils.memory import LargeReplayBuffer
from robamine.algo.util import Transition

def run():
    with open('params_iason.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = gym.make(params['env']['name'], params=params['env']['params'])

    obs = env.reset(seed=0)
    action = np.array([0, 0.9, 0, 1])
    obs_next, reward, done, _ = env.step(action)
    transition = Transition(obs, action, reward, obs_next, done)
    buffer = LargeReplayBuffer(10, 386, 4, path='/home/iason/lol.hdf5')
    for i in range(10):
        print('step:', i)
        transition.action[0] = i * 10
        buffer.store(transition)
        print('buffer size', buffer.size())
    print('buffer', buffer(2).action)
    samples = buffer.sample(4)
    print(samples[0].action)
    print(samples[1].action)
    print(samples[2].action)
    print(samples[3].action)
    print('waiting check size')
    time.sleep(5)
    buffer.clear()
    print(buffer(0))

if __name__ == '__main__':
    run()
