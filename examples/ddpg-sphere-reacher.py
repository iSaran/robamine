from rlrl_py.algo.ddpg import DDPG

import tensorflow as tf


if __name__ == '__main__':
    with tf.Session() as sess:
        agent = DDPG(sess, 'SphereReacherShapedReward-v1', exploration_noise_sigma=0.2, log_dir='/home/iason/rlrl_logs/ddpg-sphere-reacher').train(n_episodes=1000, episode_batch_size=50, render=False, episodes_to_evaluate=15, render_eval = True)
