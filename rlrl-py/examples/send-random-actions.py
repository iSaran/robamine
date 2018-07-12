import argparse
import gym
import rlrl_py
import numpy as np
#
def run(env_id):
    env = gym.make(env_id)
    env.reset()
    env.render()

    for i_episode in range(20):
        observation = env.reset()
        for t in range(1000):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            #action = np.array([0, 0])
            # print (action)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env-id', type=str, default='Floating-BHand-v0', help='The id of the gym environment to use')
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

if __name__ == '__main__':
    args = parse_args()
    run(**args)
