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

import pickle

import importlib

from robamine.utils.math import rescale_array

import copy

# Data storing primitive classes

class Transition:
    def __init__(self,
                 state=None,
                 action=None,
                 reward=None,
                 next_state=None,
                 terminal=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal

    def array(self):
        return np.array([self.state, self.action, self.reward, self.next_state, self.terminal])

    def __str__(self):
        return '[state: ' + str(self.state) + \
                ', action: ' + str(self.action) + \
                ', reward: ' + str(self.reward) + \
                ', next_state: ' + str(self.next_state) + \
                ', terminal: ' + str(self.terminal) + ']'

    def __copy__(self):
        return Transition(state=copy.copy(self.state), action=copy.copy(self.action), reward=copy.copy(self.reward), next_state=copy.copy(self.next_state), terminal=copy.copy(self.terminal))

    def copy(self):
        return self.__copy__()

class Datapoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return 'Datapoint[x= ' + str(self.x) + \
                ', y= ' + str(self.y) + ']'

    def __repr__(self):
        return  self.__str__()

class Dataset(list):
    def to_minibatches(self, batch_size=32):
        temp = self.copy()
        random.shuffle(temp)
        for _ in range(len(self) % batch_size):
            del temp[-1]
        return [Dataset(temp[i:i + batch_size]) for i in range(0, len(temp), batch_size)]

    def to_array(self):
        x = np.array([_.x for _ in self])
        y = np.array([_.y for _ in self])
        return x, y

    @classmethod
    def from_array(cls, x_array, y_array):
        cls = Dataset()
        for i in range(x_array.shape[0]):
            cls.append(Datapoint(x=x_array[i], y=y_array[i]))
        return cls

    def save(self, path, name='robamine_dataset'):
        pickle.dump(self, open(os.path.join(path, name) + '.pkl', 'wb'))

    def split(self, train_perc = 0.8):
        elements = int(len(self) * train_perc)
        train = Dataset(self[0:elements])
        test = Dataset(self[elements:])
        return train, test

    def normalize(self):
        x, y = self.to_array()
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)

        for point in range(len(self)):
            self[point].x = x[point]
            self[point].y = y[point]

    def rescale(self, ranges=[0, 1]):
        assert ranges[1] > ranges[0]
        x, y = self.to_array()
        x_rescaled = rescale_array(x, ranges, axis=0)
        y_rescaled = rescale_array(y, ranges, axis=0)

        for i in range(len(self)):
            self[i].x = x_rescaled[i]
            self[i].y = y_rescaled[i]

    def max(self):
        x, y = self.to_array()
        return np.max(x, axis=0), np.max(y, axis=0)

    def min(self):
        x, y = self.to_array()
        return np.min(x, axis=0), np.min(y, axis=0)

    def check(self):
        x, y = self.to_array()

        # Check x for nans
        if np.isnan(x).any():
            raise ValueError("Dataset: x contain NaN values")

        # Check y for nans
        if np.isnan(y).any():
            raise ValueError("Dataset: y contain NaN values")

class EnvData:
    """
    Stores data from an environment. Useful for storing samples, such as init
    states, transitions etc, drawn from an environment
    """
    def __init__(self, info_names = []):
        self.init_states = []
        self.init_observations = []
        self.transitions = []

        self.info = {}
        for i in info_names:
            self.info[i] = []

    def reset(self):
        self.init_states = []
        self.init_observations = []
        self.transitions = []
        for key in self.info:
            self.info[key] = []

    def save(self, dir, file_name='samples'):
        pickle.dump(self, open(os.path.join(dir, file_name + '.env'), 'wb'))

    @classmethod
    def load(cls, file_path):
        cls = pickle.load(open(file_path, 'rb'))
        return cls

    def __add__(self, other):
        new = EnvData()
        assert set(self.keys()) == set(other.keys())
        new.init_states = self.init_states + other.init_states
        new.init_observations = self.init_observations + other.init_observations
        new.transitions = self.transitions + other.transitions
        for key in self.info:
            new.info[key] = self.info[key] + other.info[key]
        return new

    def __iadd__(self, other):
        self.init_states += other.init_states
        self.init_observations += other.init_observations
        self.transitions += other.transitions
        for key in self.info:
            self.info[key] += other.info[key]
        return self

    def __str__(self):
        return '[init_states: ' + str(self.init_states) + \
                ', init_observations: ' + str(self.init_observations) + \
                ', transitions: ' + str(self.transitions) + \
                ', info: ' + str(self.info) + ']'

class TimestepData:
    def __init__(self):
        self.transition = None
        self.q_value = None

class EpisodeData(list):
    def __init__(self, *args):
        list.__init__(self, *args)

        self.success = False

        self.reward = {'mean': None, 'std': None, 'max': None, 'min': None, 'sum': None}
        self.q_value = {'mean': None, 'std': None, 'max': None, 'min': None, 'sum': None}
        self.actions_performed = []
        self.monte_carlo_return = []

    def calc(self):
        rewards, q_values = [], []

        for timestep in self:
            rewards.append(timestep.transition.reward)
            q_values.append(timestep.q_value)
            self.actions_performed.append(timestep.transition.action)

        self.reward['mean'] = np.mean(np.array(rewards))
        self.reward['std'] = np.std(np.array(rewards))
        self.reward['max'] = np.max(np.array(rewards))
        self.reward['min'] = np.min(np.array(rewards))
        self.reward['sum'] = np.sum(np.array(rewards))

        self.q_value['mean'] = np.mean(np.array(q_values))
        self.q_value['std'] = np.std(np.array(q_values))
        self.q_value['max'] = np.max(np.array(q_values))
        self.q_value['min'] = np.min(np.array(q_values))
        self.q_value['sum'] = np.sum(np.array(q_values))

class EpisodeListData(list):
    def __init__(self, *args):
        list.__init__(self, *args)

        self.success_rate = 0.0
        self.reward = {'mean': 0.0, 'std': 0.0}
        self.actions_success = {'mean': None, 'std': None}

    def calc(self):
        sums, nr_actions_success, successes = [], [], []
        for episode in self:
            episode.calc()
            sums.append(episode.reward['sum'])
            successes.append(float(episode.success))
            if episode.success:
                nr_actions_success.append(len(episode.actions_performed))

        self.reward['mean'] = np.mean(np.array(sums))
        self.reward['std'] = np.std(np.array(sums))
        self.actions_success['mean'] = np.mean(np.array(nr_actions_success))
        self.actions_success['std'] = np.std(np.array(nr_actions_success))
        self.success_rate = np.mean(np.array(successes))

    def save(self, dir, file_name='episodes'):
        pickle.dump(self, open(os.path.join(dir, file_name), 'wb'))

    @classmethod
    def load(cls, file_path):
        cls = pickle.load(open(file_path, 'rb'))
        return cls

    def __str__(self):
        return 'EpisodeListData:' \
               + '\n' + 'Success rate: ' + str(self.success_rate * 100) + '%' \
               + '\n' + 'Mean reward: ' + str(self.reward['mean']) \
               + '\n' + 'Std reward: ' + str(self.reward['std']) \
               + '\n' + 'Mean actions until success: ' + str(self.actions_success['mean']) \
               + '\n' + 'Std actions until success: ' + str(self.actions_success['std'])


    def dict(self):
        pass


# Primitive classes for logging data and plotting e.g. in Tensorboard

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
        self.file = open(os.path.join(self.log_path, name + '.csv'), "w+")
        # Setup the first row (the name of the logged variables)
        self.file.write(','.join(stats_name) + '\n')

        # Setup Tensorboard logging.
        self.tf_stats_name = [name + '/' + var for var in stats_name]  # the list of names for tf variables and summaries
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
        self.file.write(",".join(map(str, y)) + '\n')
        self.file.flush()

        # Parse variables to TF summaries
        feed_dict = {}
        for i in range(0, len(y)):
            feed_dict[self.tf_stats[self.tf_stats_name[i]]] = y[i]
        summary_str = self.sess.run(self.tf_summary_ops, feed_dict=feed_dict)
        self.tf_writer.add_summary(summary_str, x)
        self.tf_writer.flush()

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
        Used for logging
    name : str
        A name for these stats
    """
    def __init__(self, sess, log_dir, tf_writer, name, additional_info = {}, stats_name = ['mean', 'min', 'max', 'std', 'sum']):
        self.stats_name = stats_name
        log_var_names = ['n_timesteps', 'success']
        for j in stats_name:
            log_var_names.append(j + '_reward')
            log_var_names.append(j + '_q_value')
            for k in additional_info:
                log_var_names.append(j + '_' + k)

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

        row = [episode_data['n_timesteps'], int(episode_data['success'])]
        for operation in self.stats_name:
            operation = getattr(importlib.import_module('numpy'), operation)
            row.append(np.squeeze(operation(np.array(episode_data['reward']))))
            row.append(np.squeeze(operation(np.array(episode_data['q_value']))))
            for k in episode_data['info']:
                row.append(np.squeeze(operation(np.array(episode_data['info'][k]))))
        self.data_stream.log(episode, row)

class Plotter:
    def __init__(self, directory, streams, linewidth=1, _format='eps', dpi=1000):
        self.directory = directory
        self.streams = streams
        self.linewidth = linewidth
        self.format = _format
        self.dpi = dpi
        self.logger = logging.getLogger('robamine.algo.util.Plotter')

    def extract_var_names(self, prefixes=['mean', 'min', 'max', 'std']):
        var_names = {}
        for stream in self.streams:
            var_names[stream] = set()

            data = pd.read_csv(os.path.join(os.path.join(self.directory, stream), stream + '.log'))
            y_label = list(data.columns.values)

            for label in y_label:
                found_at_least_one_prefix = False
                for prefix in prefixes:
                    if label.startswith(prefix + '_'):
                        label_without_prefix = label[len(prefix) + 1:]
                        var_names[stream].add(label_without_prefix)
                        found_at_least_one_prefix = True
                if not found_at_least_one_prefix:
                    self.logger.warning('Label %s in %s does not have stats prefixes. I will ignore it.', label, stream)
        return var_names

    def extract_data(self, stream):
        data = pd.read_csv(os.path.join(os.path.join(self.directory, stream), stream + '.log'))
        x = list(data.index.values)
        y_label = list(data.columns.values)

        y = {}
        for i in y_label:
            y[i] = data[i]

        x_label = 'episode'
        return x, y, x_label, y_label

    def plot(self):
        y_var_label = self.extract_var_names()
        for stream in self.streams:
            x, y, x_label, y_label = self.extract_data(stream)

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

    @staticmethod
    def create_batch_from_stream(file_path, stream, batch_size):
        initial_csv_path = os.path.join(os.path.join(file_path, stream), stream + '.log')
        batch_path = os.path.join(os.path.join(file_path, 'batch_' + stream))
        if not os.path.exists(batch_path):
            os.makedirs(batch_path)

        batch = Plotter.create_batch(os.path.join(file_path, stream), stream + '.log', batch_size)
        batch.to_csv(os.path.join(batch_path, 'batch_' + stream + '.log'))

    @staticmethod
    def create_batch(file_path, file_name, batch_size, cols=['mean_reward', 'mean_q_value'], write_to_file=False):
        # Load file
        df = pd.read_csv(os.path.join(file_path, file_name))

        # Keep only the columns of interest from the log file
        for col in list(df.columns.values):
            if col not in cols:
                del df[col]

        frames = [df.groupby(df.index // 5).mean().add_prefix('mean_'),
                  df.groupby(df.index // 5).min().add_prefix('min_'),
                  df.groupby(df.index // 5).max().add_prefix('max_'),
                  df.groupby(df.index // 5).std().add_prefix('std_')]
        result =  pd.concat(frames, axis=1)
        result.index.name = 'episode_batch'
        if write_to_file:
            result.to_csv(os.path.join(file_path, 'batch_' + file_name))
            logger.info('Writing batch file to: %s', os.path.join(file_path, 'batch_' + file_name))
        return result

# Noises

class Noise:
    def __init__(self):
        self.random = np.random.RandomState()

    def seed(self, seed):
        self.random.seed(seed)

class OrnsteinUhlenbeckActionNoise(Noise):
    """
    Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
    based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """
    def __init__(self, mu, sigma = 0.2, theta=.15, dt=1e-2, x0=None, seed=999):
        super().__init__()
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * self.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class NormalNoise(Noise):
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return self.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# Misc functions

def get_agent_handle(agent_name):
    module = importlib.import_module('robamine.algo.' + agent_name.lower())
    handle = getattr(module, agent_name)
    return handle

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

def print_progress(episode, n_episodes, start_time, steps, experience_time):
    percent = (episode + 1) / n_episodes * 100.0
    time_elapsed = transform_sec_to_timestamp(time.time() - start_time)
    estimated_time = transform_sec_to_timestamp((n_episodes - episode + 1) * (time.time() - start_time) / (episode + 1))
    experience_time = transform_sec_to_timestamp(experience_time)
    logger.info('Progress: Episode: %s from %s (%.2f%%). Timesteps: %s. Time elapsed: %s. Estimated time: %s. Experience time: %s.', str(episode + 1), str(n_episodes), percent, steps, time_elapsed, estimated_time, experience_time)
    # logger.info('  Experience Time: %s', transform_sec_to_timestamp(step * dt))
