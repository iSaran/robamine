"""
Utilities for RL algorithms
===========================

This module contains helpers classes and methods for RL algorithms.
"""
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

def transform_sec_to_timestamp(seconds):
    """
    Transforms seconds to a timestamp string in format: hours:minutes:seconds

    Parameters
    ----------
    seconds : float
        The seconds to transform

    Returns
    -------
    str
        The timestamp
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

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

        self.data = {}
        self.data['episode_total_reward'] = tf.Variable(0.)
        self.data['episode_average_q'] = tf.Variable(0.)

        self.data['epoch_reward/mean'] = tf.Variable(0.)
        self.data['epoch_reward/min'] = tf.Variable(0.)
        self.data['epoch_reward/max'] = tf.Variable(0.)
        self.data['epoch_reward/std'] = tf.Variable(0.)

        for key in self.data:
            tf.summary.scalar(key, self.data[key])

        self.summary_ops = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)

    def log(self, data, x):
        feed_dict = {}
        for key in data:
            feed_dict[self.data[key]] = data[key]
        summary_str = self.sess.run(self.summary_ops, feed_dict=feed_dict)
        self.writer.add_summary(summary_str, x)
        self.writer.flush()

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

class Stats:
    """
    Compiles stats from training and evaluation.


    Example
    -------
    You can use this class in your training loop like this:

    .. code-block:: python

        stats = Stats()  # Initialize stats in general

        for episode in range(number_of_episodes):
            stats.init_for_episode()  # Initialize the stats need to be initialized at the beginning of the episode
            for time in range(episode_horizon)
                stats.init_for_timestep()  # Initialize the stats need to be initialized at the beginning of a timestep
                # do interesting stuff
                stats.update_for_timestep()  # Update the stats need to be updated at the end of a timestep
            stats.update_for_episode()  # Update the stats need to be updated at the end of an episode

    Attributes
    ----------
    agent_name : str
        The name of the agent
    env_name : str
        The name of the environment
    dt : float
        The real timestep in seconds
    current_epoch : int
        The current epoch
    current_episode : int
        The current episode
    n_episodes : int
        The total number of episodes
    current_timestep : int
        The current timestep
    episode_total_reward : float
        The total reward of the current episode

    Parameters
    ---------
    agent_name : str
        The name of the agent
    env_name : str
        The name of the environment
    n_episodes : int
        The total number of episodes
    n_epochs : The number of epochs that divide the episodes
    dt : float
        The real timestep in seconds
    logger : :class:`.Logger`
        Used for logging
    name : str
        A name for these stats
    """
    def __init__(self, agent_name, env_name, n_episodes, n_epochs, dt, logger = None, name = ""):
        # Util staff
        self.agent_name = agent_name
        self.env_name = env_name
        self.dt = dt
        self.start_time = time.time()
        self.logger = logger
        self.name = name

        # Epoch staff
        self.current_epoch = 1
        self.n_epochs = n_epochs
        assert n_episodes % n_epochs == 0
        self.n_episodes_per_epoch = int(n_episodes / n_epochs)
        self.epoch_reward = np.zeros(shape=(self.n_episodes_per_epoch))
        self.success_rate = 0
        self.current_episode_in_epoch = 0

        # Episode staff
        self.current_episode = 1
        self.n_episodes = n_episodes
        self.episode_total_reward = 0
        self.episode_average_q = 0


        # Timestep staff
        self.current_timestep = 1
        self.n_timesteps = 200

    def init_for_epoch(self):
        """
        Initialize the stats that need to be initialize at the beginning of an
        epoch.
        """
        self.epoch_reward_mean = 0
        self.epoch_max_reward = -1e6
        self.epoch_min_reward = 1e6
        self.success_rate = 0
        self.epoch_reward = np.zeros(shape=(self.n_episodes_per_epoch))
        self.current_episode_in_epoch = 0
        pass

    def init_for_episode(self):
        """
        Initialize the stats that need to be initialize at the beginning of an
        episode.
        """
        self.episode_total_reward = 0
        self.episode_average_q = 0

    def init_for_timestep(self):
        """
        Initialize the stats that need to be initialize at the beginning of a
        time step.
        """
        pass

    def update_for_epoch(self):
        """
        Update the stats that need to be updated at the end of an
        epoch.
        """
        self.success_rate /= self.n_episodes_per_epoch

        if self.logger is not None:
            data = {}
            data['epoch_reward/mean'] = np.mean(self.epoch_reward)
            data['epoch_reward/min'] = np.mean(self.epoch_reward)
            data['epoch_reward/max'] = np.std(self.epoch_reward)
            data['epoch_reward/std'] = np.max(self.epoch_reward)
            self.logger.log(data, self.current_epoch)

        self.print_header()
        self.print_progress()
        self.print(data)
        self.current_epoch += 1

    def update_for_episode(self, info, print_stats = False):
        """
        Update the stats that need to be updated at the end of an
        episode.

        Parameters
        ---------
        print_stats : bool
            True if printing stats in the console is desired at the end of the episode
        """
        self.epoch_reward[self.current_episode_in_epoch] = self.episode_total_reward
        self.episode_average_q /= self.n_timesteps

        if 'is_success' in info:
            self.success_rate += info['is_success']
        else:
            self.success_rate = float('nan')

        if self.logger is not None:
            data = {}
            data['episode_total_reward'] = self.episode_total_reward
            data['episode_average_q'] = self.episode_average_q
            self.logger.log(data, self.current_episode)

    def update_for_timestep(self, reward, q_value, current_timestep):
        """
        Update the stats that need to be updated at the end of a
        time step.

        Prameters
        ---------
        reward : float
            The reward of this timestep
        current_timestep : int
            The current timestep
        """
        self.episode_total_reward += reward
        self.episode_average_q += q_value
        self.current_timestep += 1

    def print_header(self):
        print('')
        print('===================================================')
        print('| Algorithm:', self.agent_name, ', Environment:', self.env_name)
        print('---------------------------------------------------')

    def print_progress(self):
        print('| Progress:')
        print('|   Epoch: ', self.current_epoch, 'from', self.n_epochs)
        print('|   Episode: ', self.current_episode - 1, 'from', self.n_episodes)
        print('|   Progress: ', "{0:.2f}".format((self.current_episode - 1) / self.n_episodes * 100), '%')
        print('|   Time Elapsed:', self.get_time_elapsed())
        print('|   Experience Time:', transform_sec_to_timestamp(self.current_timestep * self.dt))

    def print(self, data):
        print('|', self.name, 'stats for', self.n_episodes_per_epoch, 'episodes :')
        for key in data:
            print('| ', key, ': ', data[key])

    def get_time_elapsed(self):
            end = time.time()
            return transform_sec_to_timestamp(end - self.start_time)

