import tensorflow as tf
import logging
import robamine as rb


if __name__ == '__main__':
    with tf.Session() as sess:
        rb.rb_logging.init('/home/iason/robamine_logs/ddpg-sphere-reacher')

        agent = rb.algo.ddpg.DDPG(sess, 'SphereReacherShapedReward-v1', exploration_noise_sigma=0.2).train(n_episodes=1000, episode_batch_size=50, render=False, episodes_to_evaluate=15, render_eval = False)
