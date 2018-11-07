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

import matplotlib.pyplot as plt
import csv
import pandas as pd

from enum import Enum
import random

import logging
logger = logging.getLogger('robamine.algo.util')

import importlib

def seed_everything(random_seed):
    random.seed(random_seed)
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

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
    return "{:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds)

def print_progress(episode, n_episodes, start_time, steps, dt):
    percent = (episode + 1) / n_episodes * 100.0
    time_elapsed = transform_sec_to_timestamp(time.time() - start_time)
    estimated_time = transform_sec_to_timestamp((n_episodes - episode + 1) * (time.time() - start_time) / episode)
    experience_time = transform_sec_to_timestamp(steps * dt)
    logger.info('Progress: Episode: %s from %s (%.2f%%). Time elapsed: %s. Estimated time: %s. Experience time: %s', str(episode + 1), str(n_episodes), percent, time_elapsed, estimated_time, experience_time)
    # logger.info('  Experience Time: %s', transform_sec_to_timestamp(step * dt))

class DataStream:
    """
    Class for logging data into logfiles, saving models during training using Tensorflow.
    """
    def __init__(self, sess, directory, tf_writer, stats_name, name):
        self.sess = sess
        self.log_path = directory
        self.tf_writer = tf_writer
        self.name = name

        # Setup file logging.
        if not os.path.exists(os.path.join(self.log_path, name)):
            os.makedirs(os.path.join(self.log_path, name))
        self.file = open(os.path.join(os.path.join(self.log_path, name), name + '.log'), "w+")
        # Setup the first row (the name of the logged variables)
        self.file.write(stats_name[0])
        for i in range(1, len(stats_name)):
            self.file.write(',' + stats_name[i])
        self.file.write('\n')

        # Setup Tensorboard logging.
        self.tf_stats_name = [name + '/' + var for var in stats_name[1:]]  # the list of names for tf variables and summaries
        self.tf_stats = {}
        summaries = []
        with tf.variable_scope('robamine_data_stream_' + name):
            for var in self.tf_stats_name:
                self.tf_stats[var] = tf.Variable(0., name=var)
                summaries.append(tf.summary.scalar(var, self.tf_stats[var]))
        self.tf_summary_ops = tf.summary.merge(summaries)

    def __del__(self):
        self.file.close()

    def log(self, x, y):
        """
        Logs a new row of data into the file.

        Parameters
        ----------
        x : scalar
            The x value, e.g. the time
        y : dict
            The y values of the data, with keys the name of the variables defined by setup_stream.
        """
        # Log a row in file.
        self.file.write('%d' % x)
        for i in y:
            self.file.write(',%f' % i)
        self.file.write('\n')
        self.file.flush()

        # Parse variables to TF summaries
        feed_dict = {}
        for i in range(0, len(y)):
            feed_dict[self.tf_stats[self.tf_stats_name[i]]] = y[i]
        summary_str = self.sess.run(self.tf_summary_ops, feed_dict=feed_dict)
        self.tf_writer.add_summary(summary_str, x)
        self.tf_writer.flush()

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma = 0.2, theta=.15, dt=1e-2, x0=None, seed=999):
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
    dt : float
        The real timestep in seconds

    episode_q_mean : float
        The mean value of Q across an episode: :math:`\\frac{1}{T}\sum_{t = 0}^T Q(s(t), a(t))`

    Parameters
    ---------
    dt : float
        The real timestep in seconds
    logger : :class:`.Logger`
        Used for logging
    name : str
        A name for these stats
    """
    def __init__(self, sess, log_dir, tf_writer, name, data_name = ['reward', 'q_value'], stats_name = ['mean', 'min', 'max', 'std']):

        self.data_name = data_name
        self.stats_name = stats_name
        log_var_names = ['episode']
        for i in data_name:
            for j in stats_name:
                log_var_names.append(j + '_' + i)

        self.data_stream = DataStream(sess, log_dir, tf_writer, log_var_names, name)

    def update(self, episode, episode_data):
        """
        Update the stats that need to be updated at the end of an
        episode.

        Parameters
        ---------
        print_stats : bool
            True if printing stats in the console is desired at the end of the episode
        """
        logger.debug('Stats: Updating for episode.')

        row = []
        for i in self.data_name:
            for operation in self.stats_name:
                operation = getattr(importlib.import_module('numpy'), operation)
                row.append(np.squeeze(operation(np.array(episode_data[i]))))
        self.data_stream.log(episode, row)


class Plotter:
    def __init__(self, directory, streams, linewidth=1, _format='eps', dpi=1000):
        self.directory = directory
        self.streams = streams
        self.linewidth = linewidth
        self.format = _format
        self.dpi = dpi

    def extract_var_names(self, prefixes=['mean', 'min', 'max', 'std']):
        var_names = {}
        for stream in self.streams:
            var_names[stream] = set()

            data = pd.read_csv(os.path.join(os.path.join(self.directory, stream), stream + '.log'))
            y_label = list(data.columns.values)
            x_label = y_label[0]
            del y_label[0]

            for label in y_label:
                found_at_least_one_prefix = False
                for prefix in prefixes:
                    if label.startswith(prefix + '_'):
                        label_without_prefix = label[len(prefix) + 1:]
                        var_names[stream].add(label_without_prefix)
                        found_at_least_one_prefix = True
                if not found_at_least_one_prefix:
                    var_names[stream].add(label)
        return var_names

    def extract_data(self, stream):
        data = pd.read_csv(os.path.join(os.path.join(self.directory, stream), stream + '.log'))
        y_label = list(data.columns.values)
        x_label = y_label[0]
        del y_label[0]

        x = data[x_label]

        y = {}
        for i in y_label:
            y[i] = data[i]

        return x, y

    def plot(self):
        y_var_label = self.extract_var_names()
        for stream in self.streams:
            # TODO: use extract data instead of duplicating code
            data = pd.read_csv(os.path.join(os.path.join(self.directory, stream), stream + '.log'))
            y_label = list(data.columns.values)
            x_label = y_label[0]
            del y_label[0]

            x = data[x_label]

            y = {}
            for i in y_label:
                y[i] = data[i]

            # Plot reward
            for i in y_var_label[stream]:
                plt.plot(x, y['mean_' + i], color="#00b8e6", linewidth=self.linewidth * 1.5, label='Mean value')
                plt.plot(x, y['min_' + i], color="#ff5733", linestyle='--', linewidth=self.linewidth, label='Min value')
                plt.plot(x, y['max_' + i], color="#9CFF33", linestyle='--', linewidth=self.linewidth, label='Max value')
                plt.fill_between(x, np.array(y['mean_' + i]) - np.array(y['std_' + i]), np.array(y['mean_' + i]) + np.array(y['std_' + i]), color="#ccf5ff", label='Mean pm std')
                plt.xlabel(x_label)
                plt.ylabel(i)
                plt.xlim(x[0], x[len(x) - 1])
                plt.title(i)
                plt.grid(color='#a6a6a6', linestyle='--', linewidth=0.5*self.linewidth)
                plt.legend()
                plt.savefig(os.path.join(os.path.join(self.directory, stream), i +'.' + self.format), format=self.format, dpi=self.dpi)
                plt.clf()
