import argparse
import logging
import robamine as rm
import yaml
import gym

def run(file):
    with open("../yaml/" + file + ".yml", 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            rm.rb_logging.init(directory=params['logging_directory'], file_level=logging.INFO)
            logger = logging.getLogger('robamine')
            if params['mode'] == 'Random':
                env = gym.make(params['env']['name'], params=params['env'])
                for i_episode in range(params['train']['episodes']):
                    observation = env.reset()
                    for t in range(3000):
                        action = env.action_space.sample()
                        observation, reward, done, info = env.step(action)
                        if done:
                            print("Episode finished after {} timesteps".format(t+1))
                            break
            else:
                world = rm.World.from_dict(params)
                if params['mode'] == 'Train & Evaluate':
                    world.train_and_eval(n_episodes_to_train=params['train']['episodes'], \
                                         n_episodes_to_evaluate=params['eval']['episodes'], \
                                         evaluate_every=params['eval_every'], \
                                         save_every=params['save_every'], \
                                         print_progress_every=10, \
                                         render_train=params['train']['render'], \
                                         render_eval=params['eval']['render'])
                elif params['mode'] == 'Train':
                    world.train(n_episodes=params['train']['episodes'], \
                                print_progress_every=1, \
                                save_every=params['save_every'])
                elif params['mode'] == 'Evaluate':
                    world.evaluate(n_episodes=params['eval']['episodes'], \
                                   print_progress_every=1, \
                                   save_every=params['save_every'])
                else:
                    logger.error('The mode does not exist. Select btn Train, Train & Evaluate and Evaluate.')
        except yaml.YAMLError as exc:
            print(exc)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', type=str, default='clutter', help='The id of the gym environment to use')
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

if __name__ == '__main__':
    args = parse_args()
    run(**args)
