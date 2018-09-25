from datetime import datetime
import os
import time

import tensorflow as tf
import numpy as np

def get_now_timestamp():
    """
    Returns a timestamp for the current datetime as a string for using it in
    log file naming.
    """
    now_raw = datetime.now()
    return str(now_raw.year) + '.' + \
           '{:02d}'.format(now_raw.month) + '.' + \
           '{:02d}'.format(now_raw.day) + '.' + \
           '{:02d}'.format(now_raw.hour) + '.' \
           '{:02d}'.format(now_raw.minute) + '.' \
           '{:02d}'.format(now_raw.second) + '.' \
           '{:02d}'.format(now_raw.microsecond)

class Logger:
    """
    Class for logging data, saving models during training using Tensorflow.
    """
    def __init__(self, sess, directory, agent_name, env_name):
        self.sess = sess
        self.agent_name = agent_name
        self.env_name = env_name
        self.counter = 0

        # Create the log path
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.log_path = os.path.join(directory, 'rlrl_py_logger_' + self.agent_name.replace(" ", "_") + "_" + self.env_name.replace(" ", "_") + '_' + get_now_timestamp())
        os.makedirs(self.log_path)
        print('rlrl_py logging to directory: ', self.log_path)

        self.episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward per episode", self.episode_reward)

        self.summary_vars = {}
        self.summary_vars['episode_reward'] = self.episode_reward
        self.summary_ops = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)

        self.start = time.time()

    def log(self, data, episode):
        self.data = data
        summary_str = self.sess.run(self.summary_ops, feed_dict={self.summary_vars['episode_reward']: data})
        self.writer.add_summary(summary_str, episode)
        self.writer.flush()

    def print_console(self, episode, total_episodes, every=1):
        self.counter += 1
        if (self.counter == every):
            print('')
            print('-----------------------------')
            print('Training Agent:', self.agent_name)
            print('Episode: ', episode + 1, 'from', total_episodes, '(Progress: ', (episode + 1)/total_episodes * 100, '%)')
            print('Episode\'s reward: ', self.data)
            print('Time Elapsed:', self.get_time_elapsed())
            print('-----------------------------')
            print('')
            self.counter = 0

    def get_time_elapsed(self):
            end = time.time()
            hours, rem = divmod(end-self.start, 3600)
            minutes, seconds = divmod(rem, 60)
            return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma = 0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
