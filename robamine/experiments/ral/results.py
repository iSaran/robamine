from robamine.algo.util import EpisodeListData
import logging
import yaml
import socket
import numpy as np
import os

import matplotlib.pyplot as plt

logger = logging.getLogger('robamine')

def analyze_multiple_eval_envs(dir_, results_dir):
    exps = [
        {'name': 'SplitAC-scr', 'path': '../ral-results/env-icra/splitac-scratch', 'action_discrete': False},
        {'name': 'SplitDQN', 'path': '../ral-results/env-icra/splitdqn-3', 'action_discrete': True},
        {'name': 'SplitDQN-13', 'path': '../ral-results/env-very-hard/splitdqn', 'action_discrete': True}]
    for i in range(len(exps)):
        exps[i]['path'] = os.path.join(dir_, exps[i]['path'])
    analyze_multiple_evals(exps, results_dir, env_name='Env-ICRA')

    exps = [
        {'name': 'Random', 'path': '../ral-results/env-hard/random-cont', 'action_discrete': False},
        {'name': 'SplitAC-scr', 'path': '../ral-results/env-hard/splitac-scratch', 'action_discrete': False},
        {'name': 'SplitDQN', 'path': '../ral-results/env-hard/splitdqn', 'action_discrete': True},
        {'name': 'Push-Target', 'path': '../ral-results/env-hard/splitac-modular/push-target', 'action_discrete': False},
        {'name': 'Push-Obstacle', 'path': '../ral-results/env-hard/splitac-modular/push-obstacle', 'action_discrete': False},
        {'name': 'SplitAC-combo', 'path': '../ral-results/env-hard/splitac-modular/combo', 'action_discrete': False}]
    for i in range(len(exps)):
        exps[i]['path'] = os.path.join(dir_, exps[i]['path'])
    analyze_multiple_evals(exps, results_dir, env_name='Env-Hard')

    exps = [{'name': 'Random', 'path': '../ral-results/env-very-hard/random-cont', 'action_discrete': False},
            {'name': 'SplitAC-scr', 'path': '../ral-results/env-very-hard/splitac-scratch', 'action_discrete': False},
            {'name': 'Push-Target', 'path': '../ral-results/env-very-hard/splitac-modular/push-target', 'action_discrete': False},
            {'name': 'Push-Target-visual', 'path': '../ral-results/env-very-hard/splitac-modular/push-target-visual',
             'action_discrete': False},
            {'name': 'Yang', 'path': '../ral-results/env-very-hard/yang',
             'action_discrete': False},
            {'name': 'Combo', 'path': '../ral-results/env-very-hard/splitac-modular/combo', 'action_discrete': False}]
    for i in range(len(exps)):
        exps[i]['path'] = os.path.join(dir_, exps[i]['path'])
    analyze_multiple_evals(exps, results_dir, env_name='Env-very-hard')

    exps = [{'name': 'Random', 'path': '../ral-results/env-walls/random', 'action_discrete': False},
            {'name': 'Combo', 'path': '../ral-results/env-walls/splitac-modular/combo', 'action_discrete': False},
            {'name': 'Combo-2', 'path': '../ral-results/env-walls/with_2', 'action_discrete': False}]
    for i in range(len(exps)):
        exps[i]['path'] = os.path.join(dir_, exps[i]['path'])
    analyze_multiple_evals(exps, results_dir, env_name='Env-walls')


def analyze_multiple_evals(exps, results_dir, env_name='Metric'):
    names, dirs = [], []
    for i in range(len(exps)):
        names.append(exps[i]['name'])
        dirs.append(exps[i]['path'])

    from tabulate import tabulate
    import seaborn
    import pandas as pd
    # seaborn.set(style="whitegrid")
    seaborn.set(style="ticks", palette="pastel")


    headers = [env_name]
    column_0 = ['Valid Episodes',
               'Singulation in 5 steps for valid episodes %',
               'Singulation in 10 steps for valid episodes %',
               'Singulation in 15 steps for valid episodes %',
               'Singulation in 20 steps for valid episodes %',
               'Fallen %',
               'Max timesteps terminals %',
               'Collision terminals %',
               'Deterministic Collision terminals %',
               'Flips terminals %',
               'Empty terminals %',
               'Invalid Env before termination %',
               'Mean reward per step',
               'Mean actions for singulation',
               'Push target used %',
               'Push Obstacle used %',
               'Extra primitive used %',
               'Model trained for (timesteps)']

    percentage = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16]

    data = [None] * len(names)
    columns = [None] * len(names)
    table = []
    for i in range(len(names)):
        headers.append(names[i])
        data[i], columns[i] = analyze_eval_in_scenes(dirs[i], action_discrete=exps[i]['action_discrete'])

    for i in range(len(column_0)):
        row = []
        row.append(column_0[i])
        for j in range(len(names)):
            if i in percentage:
                row.append(columns[j][i] * 100)
            else:
                row.append(columns[j][i])

        table.append(row)
    pd.DataFrame(table, columns=headers).to_csv(os.path.join(results_dir, env_name + '.csv'), index=False)
    print('')
    print(tabulate(table, headers=headers))

    # Violin plots
    fig, axes = plt.subplots()
    df = pd.DataFrame({names[0]: data[0]})
    for i in range(1, len(names)):
        df = pd.concat([df, pd.DataFrame({names[i]: data[i]})], axis=1)

    seaborn.violinplot(data=df, bw=.4, cut=2, scale='area',
                       linewidth=1, inner='box', orient='h')
    plt.axvline(x=5, color='gray', linestyle='--')

    # seaborn.violinplot(data=data[1], palette="Set3", bw=.2, cut=2,
    #                    linewidth=3)
    # seaborn.violinplot(data=data[2], palette="Set3", bw=.2, cut=2,
    #                    linewidth=3)

    # axes.violinplot(data)
    # axes.violinplot(data, [0], points=100, widths=0.3,
    #                 showmeans=True, showextrema=True, showmedians=True)
    plt.savefig(os.path.join(results_dir, env_name + '.png'))


def analyze_eval_in_scenes(dir, action_discrete=False):
    from collections import OrderedDict
    training_timesteps = 0
    path = os.path.join(dir, 'train/episodes')
    if not os.path.exists(path):
        training_timesteps = np.nan
    else:
        training_data = EpisodeListData.load(path)
        for i in range(len(training_data)):
            training_timesteps += len(training_data[i])

    data = EpisodeListData.load(os.path.join(dir, 'eval/episodes'))
    singulations, fallens, collisions, timesteps = 0, 0, 0, 0
    steps_singulations = []
    episodes = len(data)
    rewards = []
    timestep_terminals = 0
    collision_terminals = 0
    deterministic_collision_terminals = 0
    flips = 0
    episodes_terminated = 0
    empties = 0
    push_target_used = 0
    push_obstacle_used = 0
    extra_primitive_used = 0
    invalid_env_before_termination = 0

    under = [5, 10, 15, 20]
    singulation_under = OrderedDict()
    for k in under:
        singulation_under[k] = 0

    for i in range(episodes):
        timesteps += len(data[i])
        if data[i][-1].transition.terminal:
            episodes_terminated += 1
            if data[i][-1].transition.info['termination_reason'] == 'singulation':
                for k in under:
                    if len(data[i]) <= k:
                        singulation_under[k] += 1
                singulations += 1
                steps_singulations.append(len(data[i]))
            elif data[i][-1].transition.info['termination_reason'] == 'fallen':
                fallens += 1
            elif data[i][-1].transition.info['termination_reason'] == 'timesteps':
                timestep_terminals += 1
            elif data[i][-1].transition.info['termination_reason'] == 'collision':
                collision_terminals += 1
            elif data[i][-1].transition.info['termination_reason'] == 'deterministic_collision':
                deterministic_collision_terminals += 1
            elif data[i][-1].transition.info['termination_reason'] == 'flipped':
                episodes_terminated -= 1
                flips += 1
            elif data[i][-1].transition.info['termination_reason'] == 'empty':
                empties += 1
            elif data[i][-1].transition.info['termination_reason'] == '':
                episodes_terminated -= 1
                invalid_env_before_termination += 1
            else:
                raise Exception(data[i][-1].transition.info['termination_reason'])

            for j in range(len(data[i])):
                rewards.append(data[i][j].transition.reward)

        for timestep in range(len(data[i])):
            if action_discrete:
                if data[i][timestep].transition.action < 8:
                    push_target_used += 1
                elif data[i][timestep].transition.action < 16:
                    push_obstacle_used += 1
                else:
                    extra_primitive_used += 1
            else:
                if data[i][timestep].transition.action[0] == 0:
                    push_target_used += 1
                elif data[i][timestep].transition.action[0] == 1:
                    push_obstacle_used += 1

    # print('terminal singulations:', (singulations / episodes) * 100, '%')
    # print('terminal fallens:', (fallens / episodes) * 100, '%')
    # print('collisions:', (collisions / episodes) * 100, '%')
    # print('Total timesteps:', timesteps)
    # print('Mean steps for singulation:', np.mean(steps_singulations))
    # plt.hist(steps_singulations)
    # plt.show()
    # plt.hist(rewards)
    # plt.show()
    # data = sorted(steps_singulations)
    # print('25th perc:', data[int(0.25 * len(data))])
    # print('50th perc:', data[int(0.5 * len(data))])
    # print('75th perc:', data[int(0.75 * len(data))])


    for k in under:
        singulation_under[k] /= episodes_terminated

    results = [episodes_terminated,
               singulation_under[5],
               singulation_under[10],
               singulation_under[15],
               singulation_under[20],
               (fallens / episodes),
               (timestep_terminals / episodes),
               (collision_terminals / episodes),
               (deterministic_collision_terminals / episodes),
               (flips / episodes),
               (empties / episodes),
               (invalid_env_before_termination / episodes),
               np.mean(rewards),
               np.mean(steps_singulations),
               push_target_used / timesteps,
               push_obstacle_used / timesteps,
               extra_primitive_used / timesteps,
               training_timesteps]

    return steps_singulations, results

if __name__ == '__main__':
    pid = os.getpid()
    print('Process ID:', pid)
    hostname = socket.gethostname()
    exp_dir = 'robamine_logs_dream_2020.07.02.18.35.45.636114'

    yml_name = 'params.yml'
    if hostname == 'dream':
        logging_dir = '/home/espa/robamine_logs/'
    elif hostname == 'triss':
        logging_dir = '/home/iason/robamine_logs/2020.01.16.split_ddpg/'
    elif hostname == 'iti-479':
        logging_dir = '/home/mkiatos/robamine/logs/'
    else:
        raise ValueError()
    with open(yml_name, 'r') as stream:
        params = yaml.safe_load(stream)

    # logging_dir = '/tmp'
    params['world']['logging_dir'] = logging_dir
    params['env']['params']['vae_path'] = os.path.join(logging_dir, 'VAE')

    analyze_multiple_eval_envs(params['world']['logging_dir'],
                               results_dir=os.path.join(logging_dir, '../ral-results/results'))
