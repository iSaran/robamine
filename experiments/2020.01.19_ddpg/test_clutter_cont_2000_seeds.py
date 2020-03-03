import gym
import yaml
import robamine
import numpy as np

def run():
    with open('params_dream.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = gym.make(params['env']['name'], params=params['env']['params'])

    # Push target
    # -----------
    action = np.array([0, 0.5, 0, -1])
    for i in range(1000, 5000):
        action = np.array([0, 0, -1, 1])
        env.reset(seed=i)
        env.step(action)

if __name__ == '__main__':
    run()
