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

        self.episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward per episode", self.episode_reward)

        self.summary_vars = {}
        self.summary_vars['episode_reward'] = self.episode_reward
        self.summary_ops = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)


    def log(self, data, episode):
        self.data = data
        summary_str = self.sess.run(self.summary_ops, feed_dict={self.summary_vars['episode_reward']: data})
        self.writer.add_summary(summary_str, episode)
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
    current_episode : int
        The current episode
    n_episodes : int
        The total number of episodes
    current_timestep : int
        The current timestep
    step_counter : int
        The total number of steps that has been performed in the loop
    episode_total_reward : float
        The total reward of the current episode
    agent_name : str
        The name of the agent
    env_name : str
        The name of the environment
    dt : float
        The real timestep in seconds

    Parameters
    ---------
    agent_name : str
        The name of the agent
    env_name : str
        The name of the environment
    n_episodes : int
        The total number of episodes
    dt : float
        The real timestep in seconds
    logger : :class:`.Logger`
        Used for logging
    """
    def __init__(self, agent_name, env_name, n_episodes, dt, logger = None):
        self.current_episode = 0
        self.n_episodes = n_episodes
        self.current_timestep = 0
        self.step_counter = 0
        self.episode_total_reward = 0

        self.agent_name = agent_name
        self.env_name = env_name
        self.dt = dt

        self.start_time = time.time()

        self.logger = logger

    def init_for_episode(self):
        """
        Initialize the stats that need to be initialize at the beginning of an
        episode.
        """
        self.episode_total_reward = 0

    def init_for_timestep(self):
        """
        Initialize the stats that need to be initialize at the beginning of a
        time step.
        """
        pass

    def update_for_episode(self, current_episode, print_stats = False):
        """
        Update the stats that need to be updated at the end of an
        episode.

        Prameters
        ---------
        current_episode : int
            The current episode
        print_stats : bool
            True if printing stats in the console is desired at the end of the episode
        """
        self.current_episode = current_episode

        if print_stats:
            self.print_console()

        if self.logger is not None:
            self.logger.log(self.episode_total_reward, self.current_episode)

    def update_for_timestep(self, reward, current_timestep):
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
        self.step_counter += 1
        self.current_timestep = current_timestep

    def print_console(self):
        """
        Prints basic stats in console for monitoring long training loops.
        """
        print('-----------------------------')
        print('Training Agent:', self.agent_name, 'for environment:', self.env_name)
        print('Episode: ', self.current_episode + 1, 'from', self.n_episodes, '(Progress: ', (self.current_episode + 1) / self.n_episodes * 100, '%)')
        print('Episode\'s reward: ', self.episode_total_reward)
        print('Time Elapsed:', self.get_time_elapsed())
        print('Experience Time:', transform_sec_to_timestamp(self.step_counter * self.dt))
        print('-----------------------------')
        print('')

    def get_time_elapsed(self):
            end = time.time()
            return transform_sec_to_timestamp(end - self.start_time)

