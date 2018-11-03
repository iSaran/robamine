import tensorflow as tf
import logging
import robamine as rb

if __name__ == '__main__':

    rb.rb_logging.init('/home/iason/robamine_logs/ddpg-pendulum')

    with tf.Session() as sess:
        agent = rb.algo.ddpg.DDPG(sess, 'Pendulum-v0', exploration_noise_sigma=0.2)
        agent.train(n_episodes=2, episode_batch_size=1, render=True, episodes_to_evaluate=1, render_eval = True)
