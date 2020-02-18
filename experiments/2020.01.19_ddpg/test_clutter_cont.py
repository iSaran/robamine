import gym
import yaml
import robamine
import numpy as np

def run():
    with open('params.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = gym.make(params['env']['name'], params=params['env']['params'])
    env.seed(0)

    # Push target
    # -----------
    action = np.array([0, 0.5, 0, 0.2])
    env.reset()
    env.step(action)

    # Push obstacle
    # -------------
    action = np.array([1, 1, 0, 0.2])
    env.reset()
    env.step(action)

if __name__ == '__main__':
    run()
