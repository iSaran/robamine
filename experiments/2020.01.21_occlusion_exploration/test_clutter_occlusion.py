import gym
import yaml
import robamine
import numpy as np
import pickle

def run():
    with open('params.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = gym.make(params['env']['name'], params=params['env']['params'])

    action = np.array([0, 1, 0, 0.2])
    env.reset()
    env.step(action)

    with open('log.pkl', 'wb') as stream:
        pickle.dump(env.log, stream)

if __name__ == '__main__':
    run()
