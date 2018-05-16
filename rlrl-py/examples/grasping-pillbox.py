import gym
import rlrl_py
import numpy as np
#
env = gym.make('Floating-BHand-v0')
env.reset()
env.render()

for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        # action = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        print (action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break



