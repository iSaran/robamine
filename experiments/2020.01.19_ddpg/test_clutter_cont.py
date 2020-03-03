import gym
import yaml
import robamine
import numpy as np

def run():
    with open('params_dream.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = gym.make(params['env']['name'], params=params['env']['params'])

    rng = np.random.RandomState()
    for i in range(285, 5000):
    # for i in range(1):
        print('seed: ', i)
        # i = 282
        env.reset(seed=i)
        rng.seed(i)
        angle = rng.uniform(-1, 1)
        print('angle', angle)
        action = np.array([0, 0.5, 0, 1])
        try:
            env.step(action)
        except:
            print('failed to step moving on')
            continue

    # # Push target
    # # -----------
    # env.reset(seed=0)
    # # env.reset(seed=0)
    # env.step(action)
    #
    # action = np.array([0, 0.5, -1, 0.2])
    # # env.reset(seed=0)
    # env.step(action)
    #
    # # Push obstacle
    # # -------------
    # action = np.array([1, 1, 0, 1])
    # env.reset()
    # env.step(action)

if __name__ == '__main__':
    run()
