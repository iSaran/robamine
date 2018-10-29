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

from enum import Enum

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
    Class for logging data into logfiles, saving models during training using Tensorflow.
    """
    def __init__(self, sess, directory, agent_name, env_name):
        self.sess = sess
        self.agent_name = agent_name
        self.env_name = env_name

        # Create the log path
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.log_path = os.path.join(directory, 'rlrl_py_logger_' + self.agent_name.replace(" ", "_") + "_" + self.env_name.replace(" ", "_") + '_' + get_now_timestamp())
        os.makedirs(self.log_path)

        self.console = Console(self.log_path)

        self.console.info('Time and date: ' + str(datetime.now()))
        self.console.info('Agent: ' + self.agent_name)
        self.console.info('Environment: ' + self.env_name)
        self.console.info('Logging directory: ' + self.log_path)

        # Create the log file
        self.file = {}

        # Initialize variable for storing stats names for logging
        self.stats = {}
        self.tf_stats = {}
        self.tf_summary_ops = {}

        self.tf_writer = tf.summary.FileWriter(self.log_path, self.sess.graph)

    def flush(self):
        for i in self.file:
            self.file[i].flush()
        self.console.file.flush()


    def setup_stream(self, stream, stats):
        """
        Set up a stream. Basically writes the names of the variables to be
        logged in the first row of the log file, defined by a stream. It should
        be called before the log function.

        Parameters
        ---------
        stream : str
            The name of the stream
        stats : list
            A list of name variables to be logged.
        """
        if not os.path.exists(os.path.join(self.log_path, stream)):
            os.makedirs(os.path.join(self.log_path, stream))
        self.file[stream] = open(os.path.join(os.path.join(self.log_path, stream), stream + '.log'), "w+")

        self.tf_stats[stream] = {}

        self.stats[stream] = stats

        # Setup the first row (the name of the logged variables)
        self.file[stream].write(stream)
        for i in self.stats[stream]:
            self.file[stream].write(',' + i)
        self.file[stream].write('\n')

        # Log for Tensorboard
        for variable_name in self.stats[stream]:
            self.tf_stats[stream][stream + '/' + variable_name] = tf.Variable(0.)
        summaries = []
        for tf_variable_name in self.tf_stats[stream]:
            summaries.append(tf.summary.scalar(tf_variable_name, self.tf_stats[stream][tf_variable_name]))
        self.tf_summary_ops[stream] = tf.summary.merge(summaries)

    def __del__(self):
        for key in self.file:
            self.file[key].close()

    def log(self, x, y, stream):
        """
        Logs a new row of data into the file.

        Parameters
        ----------
        x : scalar
            The x value, e.g. the time
        y : dict
            The y values of the data, with keys the name of the variables defined by setup_stream.
        """
        # Log a row.
        self.file[stream].write('%d' % x)
        for i in self.stats[stream]:
            self.file[stream].write(',%f' % y[i])
        self.file[stream].write('\n')

        feed_dict = {}
        for var_name in y:
            feed_dict[self.tf_stats[stream][stream + '/' + var_name]] = y[var_name]
        summary_str = self.sess.run(self.tf_summary_ops[stream], feed_dict=feed_dict)
        self.tf_writer.add_summary(summary_str, x)
        self.tf_writer.flush()

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma = 0.2, theta=.15, dt=1e-2, x0=None):
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
    def __init__(self, dt, logger = None, timestep_stats = ['reward', 'q_value'], episode_stats = {'reward': ['mean', 'min', 'max', 'std'], 'q_value': ['mean', 'min', 'max', 'std']}, batch_stats = {'mean_reward': ['mean', 'min', 'max', 'std'], 'mean_q_value': ['mean', 'min', 'max', 'std']}, name = ""):
        # Util staff
        self.dt = dt
        self.start_time = time.time()
        self.logger = logger
        self.name = name

        self.step = 0

        self.timestep_stats = timestep_stats
        self.episode_stats = episode_stats
        self.batch_stats = batch_stats

        self.episode_stats_names = []
        self.batch_stats_names = []
        self.timestep_data = {}
        self.episode_data = {}
        for i in self.timestep_stats:
            self.timestep_data[i] = []

        for episode_stat in self.episode_stats:
            for operation in self.episode_stats[episode_stat]:
                self.episode_stats_names.append(operation + '_' + episode_stat)
                self.episode_data[operation + '_' + episode_stat] = []

        if self.batch_stats is not None:
            for batch_stat in self.batch_stats:
                for operation in self.batch_stats[batch_stat]:
                    self.batch_stats_names.append(operation + '_' + batch_stat)

        self.logger.setup_stream(self.name + '_episode', self.episode_stats_names)

        if self.batch_stats is not None:
            self.logger.setup_stream(self.name + '_batch', self.batch_stats_names)

    def perform_operation(self, data, operation):
        if operation == 'total':
            return np.squeeze(np.sum(np.array(data)))
        elif operation == 'mean':
            return np.squeeze(np.mean(np.array(data)))
        elif operation == 'min':
            return np.squeeze(np.min(np.array(data)))
        elif operation == 'max':
            return np.squeeze(np.max(np.array(data)))
        elif operation == 'std':
            return np.squeeze(np.std(np.array(data)))
        else:
            raise ValueError('The operation is not valid. Valid operations are total, mean, min, max, std')

    def update_timestep(self, data):
        """
        Update the stats that need to be updated at the end of a
        time step.

        Parameters
        ---------
        reward : float
            The reward of this timestep
        current_timestep : int
            The current timestep
        """
        for i in self.timestep_stats:
            self.timestep_data[i].append(data[i])

        self.step += 1

    def update_episode(self, episode):
        """
        Update the stats that need to be updated at the end of an
        episode.

        Parameters
        ---------
        print_stats : bool
            True if printing stats in the console is desired at the end of the episode
        """

        data_log = {}
        for stat in self.timestep_stats:
            for operation in self.episode_stats[stat]:
                data_log[operation + '_' + stat] = self.perform_operation(self.timestep_data[stat], operation)
                self.episode_data[operation + '_' + stat].append(data_log[operation + '_' + stat])

        if self.logger is not None:
            self.logger.log(episode, data_log, stream=self.name + '_episode')

        # Initilize staff for the next loop
        for i in self.timestep_stats:
            self.timestep_data[i] = []

    def update_batch(self, batch):
        """
        Update the stats that need to be updated at the end of an
        epoch.
        """
        if self.batch_stats is None:
            self.logger.console.warn('update_batch(): Batch stats are None, nothing to do.')
            return

        data_log = {}
        for stat in self.batch_stats:
            for operation in self.batch_stats[stat]:
                data_log[operation + '_' + stat] = self.perform_operation(self.episode_data[stat], operation)

        if self.logger is not None:
            self.logger.log(batch, data_log, stream=self.name + '_batch')

        # Initilize staff for the next loop
        for i in self.timestep_stats:
            self.timestep_data[i] = []

        for episode_stat in self.episode_stats:
            for operation in self.episode_stats[episode_stat]:
                self.episode_data[operation + '_' + episode_stat] = []

    def print_progress(self, agent_name, env_name, episode, n_episodes):
        self.logger.console.info('')
        self.logger.console.info('===================================================')
        self.logger.console.info('| Algorithm: ' + agent_name + ', Environment:' + env_name)
        self.logger.console.info('---------------------------------------------------')
        self.logger.console.info('| Progress:')
        self.logger.console.info('|   Episode: ' + str(episode) + ' from ' + str(n_episodes))
        self.logger.console.info('|   Progress: ' + "{0:.2f}".format(episode / n_episodes * 100) + '%')
        self.logger.console.info('|   Time Elapsed: ' + self.get_time_elapsed())
        self.logger.console.info('|   Experience Time: ' + transform_sec_to_timestamp(self.step * self.dt))
        self.logger.console.info('===================================================')

    def get_time_elapsed(self):
        end = time.time()
        return transform_sec_to_timestamp(end - self.start_time)

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

            with open(os.path.join(os.path.join(self.directory, stream), stream + '.log'), 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                y_label = next(reader)
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

    def plot(self):
        y_var_label = self.extract_var_names()
        for stream in self.streams:
            x = []
            y = {}
            with open(os.path.join(os.path.join(self.directory, stream), stream + '.log'), 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                y_label = next(reader)
                x_label = y_label[0]
                del y_label[0]

                for i in y_label:
                    y[i] = []
                for row in reader:
                    x.append(int(row[0]))
                    for i in range(len(row) - 1):
                        y[y_label[i]].append(float(row[i + 1]))

            # Plot reward
            for i in y_var_label[stream]:
                plt.plot(x, y['mean_' + i], color="#00b8e6", linewidth=self.linewidth * 1.5, label='Mean value')
                plt.plot(x, y['min_' + i], color="#ff5733", linestyle='--', linewidth=self.linewidth, label='Min value')
                plt.plot(x, y['max_' + i], color="#9CFF33", linestyle='--', linewidth=self.linewidth, label='Max value')
                plt.fill_between(x, np.array(y['mean_' + i]) - np.array(y['std_' + i]), np.array(y['mean_' + i]) + np.array(y['std_' + i]), color="#ccf5ff", label='Mean pm std')
                plt.xlabel(x_label)
                plt.ylabel(i)
                plt.grid(color='#a6a6a6', linestyle='--', linewidth=0.5*self.linewidth)
                plt.legend()
                plt.savefig(os.path.join(os.path.join(self.directory, stream), i +'.' + self.format), format=self.format, dpi=self.dpi)
                plt.clf()

class OrderedEnum(Enum):
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class Verbosity(OrderedEnum):
    no = 0
    warn = 1
    info = 2
    debug = 3

class ConsoleColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    DEBUG = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Console:
    def __init__(self, directory, console = True):
        self.verbosity_level = Verbosity.info
        self.console = console
        self.prefix = '[rlrl_py]'

        self.file = open(os.path.join(directory, 'console.log'), "w+")

    def __del__(self):
        self.file.close()

    def info(self, string):

        if self.verbosity_level == Verbosity.no:
            return

        prefix = self.prefix + '[info]'

        self.file.write(prefix + " " + string + '\n')

        if self.console and self.verbosity_level >= Verbosity.info:
            print(prefix + " " + string)

    def debug(self, string):

        if self.verbosity_level == Verbosity.no:
            return

        prefix = self.prefix + '[debug]'

        self.file.write(prefix + " " + string + '\n')

        if self.console and self.verbosity_level >= Verbosity.debug:
            print(ConsoleColors.DEBUG + prefix + " " + string + ConsoleColors.ENDC)

    def warn(self, string):
        if self.verbosity_level == Verbosity.no:
            return

        prefix = self.prefix + '[warn]'

        self.file.write(prefix + " " + string + '\n')

        if self.console and self.verbosity_level >= Verbosity.warn:
            print(ConsoleColors.WARNING + ConsoleColors.BOLD + prefix + " " + string + ConsoleColors.ENDC)

    def error(self, string, raise_exception = True):
        if self.verbosity_level == Verbosity.no:
            return

        prefix = self.prefix + '[error]'

        self.file.write(prefix + " " + string + '\n')

        if self.console and self.verbosity_level >= Verbosity.warn:
            print(ConsoleColors.FAIL + ConsoleColors.BOLD + prefix + " " + string + ConsoleColors.ENDC)

        if raise_exception:
            raise RuntimeError(string)

    def set_verbosity_level(self, verbosity_level):
        self.verbosity_level = verbosity_level

