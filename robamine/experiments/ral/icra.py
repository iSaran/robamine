from robamine.algo.core import TrainWorld, EvalWorld, TrainEvalWorld, SupervisedTrainWorld
from robamine import rb_logging
import logging
import yaml
import socket
import numpy as np
import os
import gym
import matplotlib.pyplot as plt
logger = logging.getLogger('robamine')

# Train ICRA
# ----------

def icra_check_transition(params):
    from robamine.envs.clutter_cont import ClutterContICRAWrapper
    params['env']['params']['render'] = True
    params['env']['params']['push']['predict_collision'] = False
    params['env']['params']['safe'] = False
    params['env']['params']['icra']['use'] = True
    env = ClutterContICRAWrapper(params['env']['params'])

    while True:
        seed = np.random.randint(100000000)
        # seed = 36263774
        # seed = 48114142
        # seed = 86177553
        print('Seed:', seed)
        rng = np.random.RandomState()
        rng.seed(seed)
        obs = env.reset(seed=seed)

        while True:
            for action in np.arange(0, 24, 1):
                # action = 8
                print('icra discrete action', action)
                obs, reward, done, info = env.step(action)
                print('reward: ', reward, 'done:', done)
            if done:
                break

def train_icra(params):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='icra', file_level=logging.INFO)
    from robamine.algo.splitdqn import SplitDQN

    from robamine.envs.clutter_cont import ClutterContICRAWrapper
    params['env']['params']['render'] = True
    params['env']['params']['safe'] = True
    params['env']['params']['icra']['use'] = True
    env = gym.make('ClutterContICRAWrapper-v0', params=params['env']['params'])

    agent = SplitDQN(state_dim=264 * 8, action_dim=8 * 2,
                     params={'replay_buffer_size': 1e6,
                             'batch_size': [64, 64],
                             'discount': 0.9,
                             'epsilon_start': 0.9,
                             'epsilon_end': 0.05,
                             'epsilon_decay': 20000,
                             'learning_rate': [1e-3, 1e-3],
                             'tau': 0.999,
                             'double_dqn': True,
                             'hidden_units': [[100, 100], [100, 100]],
                             'loss': ['mse', 'mse'],
                             'device': 'cpu',
                             'load_nets': '',
                             'load_buffers': '',
                             'update_iter': [1, 1, 5]
                             })
    trainer = TrainWorld(agent=agent, env=env, params=params['world']['params'])
    trainer.run()

def train_eval_icra(params):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='icra_splitdqn13', file_level=logging.INFO)
    from robamine.algo.splitdqn import SplitDQN

    from robamine.envs.clutter_cont import ClutterContICRAWrapper
    params['env']['params']['render'] = False
    params['env']['params']['safe'] = True
    params['env']['params']['icra']['use'] = True
    params['env']['params']['obstacle']['pushable_threshold_coeff'] = 1
    env = gym.make('ClutterContICRAWrapper-v0', params=params['env']['params'])

    agent = SplitDQN(state_dim=263 * 8, action_dim=8 * 3,
                     params={'replay_buffer_size': 1e6,
                             'batch_size': [64, 64, 64],
                             'discount': 0.9,
                             'epsilon_start': 0.9,
                             'epsilon_end': 0.25,
                             'epsilon_decay': 20000,
                             'learning_rate': [1e-3, 1e-3, 1e-3],
                             'tau': 0.999,
                             'double_dqn': True,
                             'hidden_units': [[140, 140], [140, 140], [140, 140]],
                             'loss': ['mse', 'mse', 'mse'],
                             'device': 'cpu',
                             'load_nets': '',
                             'load_buffers': '',
                             'update_iter': [1, 1, 1],
                             'n_preloaded_buffer': [500, 500, 500]
                             })
    trainer = TrainEvalWorld(agent=agent, env=env,
                             params={'episodes': 10000,
                                     'eval_episodes': 20,
                                     'eval_every': 100,
                                     'eval_render': False,
                                     'save_every': 100})
    trainer.seed(0)
    trainer.run()

def eval_random_actions_icra(params, n_scenes=1000):
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name='', file_level=logging.INFO)

    params['env']['params']['render'] = False
    params['env']['params']['safe'] = True
    params['env']['params']['hardcoded_primitive'] = -1
    params['env']['params']['log_dir'] = params['world']['logging_dir']
    params['world']['episodes'] = n_scenes
    env = gym.make('ClutterContICRAWrapper-v0', params=params['env']['params'])

    policy = RandomICRAPolicy()
    world = EvalWorld(agent=policy, env=env, params=params['world'])
    world.seed(0)
    world.run()
    print('Logging dir:', world.log_dir)

def util_test_generation_target(params, samples=1000, bins=20):
    rng = np.random.RandomState()
    finger_size = params['finger']['size']
    objects = np.zeros((samples, 3))
    for i in range(samples):
        a = max(params['target']['min_bounding_box'][2], finger_size)
        b = params['target']['max_bounding_box'][2]
        if a > b:
            b = a
        target_height = rng.uniform(a, b)

        # Length is 0.75 at least of height to reduce flipping
        a = max(0.75 * target_height, params['target']['min_bounding_box'][0])
        b = params['target']['max_bounding_box'][0]
        if a > b:
            b = a
        target_length = rng.uniform(a, b)

        a = max(0.75 * target_height, params['target']['min_bounding_box'][1])
        b = min(target_length, params['target']['max_bounding_box'][1])
        if a > b:
            b = a
        target_width = rng.uniform(a, b)

        objects[i, 0] = target_length
        objects[i, 1] = target_width
        objects[i, 2] = target_height

    fig, axs = plt.subplots(3,)
    for i in range(3):
        axs[i].hist(objects[:, i], bins=bins)
    plt.show()

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

    # ICRA comparison
    # ---------------

    # icra_check_transition(params)
    # train_icra(params)
    train_eval_icra(params)
    # eval_random_actions_icra(params, n_scenes=1000)

