from robamine.algo.util import EpisodeListData
from robamine.algo.core import EvalWorld
import pickle
import yaml
import os
from robamine import rb_logging
import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

PRIMITIVES = ['push_target', 'push_obstacle', 'grasp_target', 'grasp_obstacle']

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def eval_with_render(dir):
    rb_logging.init(directory='/tmp/robamine_logs', friendly_name='', file_level=logging.INFO)
    trainer = EvalWorld.load(dir)
    trainer.run()

def process_eval_episodes(dir, smoothing = 0.98):
    with open(os.path.join(dir, 'config.yml'), 'r') as stream:
        config = yaml.safe_load(stream)

    session = pickle.load(open(os.path.join(dir, 'episode_stats_eval.pkl'), 'rb'))
    eval_episodes = config['world']['eval_episodes']
    n_episodes = len(session)
    n_primitives = len(config['agent']['params']['actions'])
    primitives = []
    for i in range(n_primitives):
        primitives.append(PRIMITIVES[i])


    n_timesteps_total = 0
    n_timesteps = []
    success = []
    reward, q_value = [], []
    primitive, actor_loss, critic_loss = [], [], []
    for i in range(n_primitives):
        primitive.append([])
        actor_loss.append([])
        critic_loss.append([])
    for episode in session:
        n_timesteps.append(episode['n_timesteps'])
        n_timesteps_total += n_timesteps[-1]
        success.append(int(episode['success']))
        for t in range(n_timesteps[-1]):
            reward.append(episode['reward'][t])
            q_value.append(episode['q_value'][t])
            for i in range(n_primitives):
                actor_loss[i].append(episode['info']['actor_' + str(i) + '_loss'][t])
                critic_loss[i].append(episode['info']['critic_' + str(i) + '_loss'][t])
                if int(episode['actions_performed'][t][0]) == i:
                    primitive[i].append(1)
                else:
                    primitive[i].append(0)

    fig, axs = plt.subplots(3, 3, sharex = False, sharey = False)
    axs = axs.ravel()

    # add key: smoothing if you want different smoothing for a plot
    plots = [{'signal': success, 'title': 'Success', 'per_episode': True, 'legend': None},
             {'signal': n_timesteps, 'title': 'Number of timesteps', 'per_episode': True, 'legend': None},
             {'signal': reward, 'title': 'Reward', 'per_episode': False, 'legend': None},
             {'signal': q_value, 'title': 'Q value', 'per_episode': False, 'legend': None},
             {'signal': primitive, 'title': 'Primitive used', 'per_episode': False, 'legend': primitives},
             {'signal': actor_loss, 'title': 'Actor loss', 'per_episode': False, 'legend': primitives},
             {'signal': critic_loss, 'title': 'Critic loss', 'per_episode': False, 'legend': primitives},
             ]

    for i in range(len(plots)):
        if plots[i]['per_episode']:
            x = np.arange(0, n_episodes, 1)
            xlabel = 'Episodes'
        else:
            x = np.arange(0, n_timesteps_total, 1)
            xlabel = 'Timesteps'

        if plots[i]['legend']:
            for j in range(len(plots[i]['legend'])):
                axs[i].plot(x, smooth(plots[i]['signal'][j], plots[i].get('smoothing', smoothing)))
            axs[i].legend(plots[i]['legend'])
        else:
            axs[i].plot(x, smooth(plots[i]['signal'], plots[i].get('smoothing', smoothing)))

        axs[i].set_title(plots[i]['title'])
        axs[i].set_xlabel(xlabel)
    plt.show()

if __name__ == '__main__':
    dir = '/home/iason/Dropbox/projects/phd/clutter/training/2020.01.16.split_ddpg/robamine_logs_2020.02.05.15.20.58.472410'
    process_eval_episodes(dir)
    # process_eval_episodes(dir)
    # eval_with_render(dir)
