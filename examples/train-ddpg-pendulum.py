import tensorflow as tf
import logging
import robamine as rb

if __name__ == '__main__':

    rb.rb_logging.init('/home/iason/robamine_logs/ddpg-pendulum')

    with tf.Session() as sess:
        agent = rb.algo.ddpg.DDPG(sess, 'Pendulum-v0', exploration_noise_sigma=0.2)
        agent.train(n_episodes=1000, episode_batch_size=25, render=False, episodes_to_evaluate=10, render_eval=False)
