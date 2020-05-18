import gym
import yaml
import robamine
import numpy as np

from robamine.utils.orientation import quat2rot


def run():
    with open('params.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = gym.make(params['env']['name'], params=params['env']['params'])

    # Push target
    # -----------
    action = np.array([0, -0.5, 1, -1])
    # env.reset()
    env.reset()
    env.step(action)

    action = np.array([0, -0.5, 0, -1])
    # # env.reset(seed=0)
    env.step(action)
    #
    # action = np.array([0, 1, 0, 1])
    # # env.reset()
    # env.step(action)

    action = np.array([0, -0.5, 0, 1])
    # env.reset()
    env.step(action)


if __name__ == '__main__':
    run()
