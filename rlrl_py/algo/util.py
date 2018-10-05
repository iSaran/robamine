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

        # Create the variables and the operation for the Tensorflow summaries
        self.episode_data = {}
        self.episode_data['episode_reward'] = tf.Variable(0.)
        episode_summaries = []
        for key in self.episode_data:
            episode_summaries.append(tf.summary.scalar(key, self.episode_data[key]))
        self.episode_summary_ops = tf.summary.merge(episode_summaries)

        self.epoch_data = {}
        self.epoch_data['epoch_reward/mean'] = tf.Variable(0.)
        self.epoch_data['epoch_reward/min'] = tf.Variable(0.)
        self.epoch_data['epoch_reward/max'] = tf.Variable(0.)
        self.epoch_data['epoch_reward/std'] = tf.Variable(0.)
        epoch_summaries = []
        for key in self.epoch_data:
            epoch_summaries.append(tf.summary.scalar(key, self.epoch_data[key]))
        self.epoch_summary_ops = tf.summary.merge(epoch_summaries)

        self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)

    def log_episode(self, data, x):
        feed_dict = {}
        for key in data:
            feed_dict[self.episode_data[key]] = data[key]
        summary_str = self.sess.run(self.episode_summary_ops, feed_dict=feed_dict)
        self.writer.add_summary(summary_str, x)
        self.writer.flush()

    def log_epoch(self, data, x):
        feed_dict = {}
        for key in data:
            feed_dict[self.epoch_data[key]] = data[key]
        summary_str = self.sess.run(self.epoch_summary_ops, feed_dict=feed_dict)
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
    Compiles stats from training and evaluation. The stats (the attributes of
    these stats) are named based on this convention: level_metric_semantics,
    where level is the epoch, episode, timestep, metric is the type of metric
    (e.g. reward, q value etc.) and semantics is the what kind of post
    processing for this metric has happened, e.g. mean (for the average), min
    for minimum etc.


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

    n_episodes : int
        The total number of episodes :math:`N` for an epoch
    n_epochs : int
        The total number of epochs :math:`M`

    episode : int
        The current episode :math:`n \in [0, 1, ..., N]`, during an epoch
    epoch : int
        The current epoch :math:`m \in [0, 1, ..., M]`, during a training process

    episode_reward : float
        The total reward of the current episode: :math:`\sum_{t = 0}^T r(t)`
    episode_q_mean : float
        The mean value of Q across an episode: :math:`\\frac{1}{T}\sum_{t = 0}^T Q(s(t), a(t))`

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

        self.n_episodes = n_episodes
        assert n_episodes % n_epochs == 0
        self.n_episodes_per_epoch = int(n_episodes / n_epochs)

        self.episode = 0
        self.episode_reward = []

        self.epoch = 0
        self.epoch_reward = np.empty(self.n_episodes_per_epoch)

    def update_for_timestep(self, reward, q_value):
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
        self.episode_reward.append(reward)

    def update_for_episode(self, info, print_stats = False):
        """
        Update the stats that need to be updated at the end of an
        episode.

        Parameters
        ---------
        print_stats : bool
            True if printing stats in the console is desired at the end of the episode
        """

        episode_reward_sum = np.sum(np.array(self.episode_reward))
        if self.logger is not None:
            data = {}
            data['episode_reward'] = episode_reward_sum
            self.logger.log_episode(data, self.episode + self.epoch * self.n_episodes_per_epoch)

        self.epoch_reward[self.episode] = episode_reward_sum

        # Initilize staff for the next loop
        self.episode_reward = []
        self.episode += 1

    def update_for_epoch(self):
        """
        Update the stats that need to be updated at the end of an
        epoch.
        """
        if self.logger is not None:
            data = {}
            data['epoch_reward/mean'] = np.mean(self.epoch_reward)
            data['epoch_reward/std'] = np.std(self.epoch_reward)
            data['epoch_reward/min'] = np.min(self.epoch_reward)
            data['epoch_reward/max'] = np.max(self.epoch_reward)
            self.logger.log_epoch(data, self.epoch)

        # Initilize staff for the next loop
        self.epoch_reward = np.empty(self.n_episodes_per_epoch)
        self.epoch += 1
        self.episode = 0

    def print_header(self):
        print('')
        print('===================================================')
        print('| Algorithm:', self.agent_name, ', Environment:', self.env_name)
        print('---------------------------------------------------')

    def print_progress(self):
        print('| Progress:')
        print('|   Epoch: ', self.epoch, 'from', self.n_epochs)
        print('|   Episode: ', self.episode - 1, 'from', self.n_episodes)
        print('|   Progress: ', "{0:.2f}".format((self.episode - 1) / self.n_episodes * 100), '%')
        print('|   Time Elapsed:', self.get_time_elapsed())
        print('|   Experience Time:', transform_sec_to_timestamp(self.step * self.dt))

    def print(self, data):
        print('|', self.name, 'stats for', self.n_episodes_per_epoch, 'episodes :')
        for key in data:
            print('| ', key, ': ', data[key])

    def get_time_elapsed(self):
            end = time.time()
            return transform_sec_to_timestamp(end - self.start_time)

