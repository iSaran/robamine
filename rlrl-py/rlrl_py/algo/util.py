from datetime import datetime
import os
import tensorflow as tf
import time

def get_now_timestamp():
        now_raw = datetime.now()
        return str(now_raw.year) + '.' + \
               '{:02d}'.format(now_raw.month) + '.' + \
               '{:02d}'.format(now_raw.day) + '.' + \
               '{:02d}'.format(now_raw.hour) + '.' \
               '{:02d}'.format(now_raw.minute) + '.' \
               '{:02d}'.format(now_raw.second) + '.' \
               '{:02d}'.format(now_raw.microsecond)
class Logger:
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

        self.summary_vars = [self.episode_reward]
        self.summary_ops = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)

        self.start = time.time()

    def log(self, data, episode):
        self.data = data
        summary_str = self.sess.run(self.summary_ops, feed_dict={self.summary_vars[0]: data})
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

