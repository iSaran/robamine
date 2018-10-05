from rlrl_py.algo.ddpg import DDPG

import tensorflow as tf


if __name__ == '__main__':
    with tf.Session() as sess:
        agent = DDPG(sess, 'Pendulum-v0', exploration_noise_sigma=0.01, log_dir='/home/iason/rlrl_logs/ddpg-pendulum').train(n_episodes=1000, n_epochs=100, render=False, episodes_to_evaluate=0)
