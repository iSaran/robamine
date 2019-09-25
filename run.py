import argparse
import logging
import robamine as rm
import yaml
import gym

from tensorboard import default
from tensorboard import program

from robamine.gui.gui import RobamineGUI
from PyQt5.QtWidgets import QApplication
import sys

def run(no_gui, yml):
    if no_gui:
        with open("../yaml/" + yml + ".yml", 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                rm.rb_logging.init(directory=params['world']['logging_directory'], file_level=logging.INFO)
                logger = logging.getLogger('robamine')
                if params['world']['mode'] == 'Random':
                    env = gym.make(params['env']['name'], params=params['env'])
                    for i_episode in range(params['world']['episodes']):
                        observation = env.reset()
                        for t in range(3000):
                            action = env.action_space.sample()
                            observation, reward, done, info = env.step(action)
                            if done:
                                print("Episode finished after {} timesteps".format(t+1))
                                break
                else:
                    if params['world']['load'] != '':
                        world = rm.World.load(params['world']['load'])
                    else:
                        world = rm.World.from_dict(params)

                    # Start tensorboard server
                    # logging.getLogger('tensorflow').setLevel(logging.ERROR)
                    # tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
                    # tb.configure(argv=[None, '--logdir', world.log_dir])
                    # url = tb.launch()
                    # logger.info('TensorBoard plots at %s' % url)

                    if params['world']['mode'] == 'Train & Evaluate':
                        world.train_and_eval(n_episodes_to_train=params['world']['episodes'], \
                                             n_episodes_to_evaluate=params['world']['eval']['episodes'], \
                                             evaluate_every=params['world']['eval']['every'], \
                                             save_every=params['world']['save_every'], \
                                             print_progress_every=10, \
                                             render_train=params['world']['render'], \
                                             render_eval=params['world']['eval']['render'])
                    elif params['world']['mode'] == 'Train':
                        world.train(n_episodes=params['world']['episodes'], \
                                    render=params['world']['render'], \
                                    print_progress_every=10, \
                                    save_every=params['save_every'])
                    elif params['world']['mode'] == 'Evaluate':
                        world.evaluate(n_episodes=params['world']['episodes'], \
                                       render=params['world']['render'], \
                                       print_progress_every=10,
                                       save_every=params['save_every'])
                    else:
                        logger.error('The mode does not exist. Select btn Train, Train & Evaluate and Evaluate.')
            except yaml.YAMLError as exc:
                print(exc)

    else:
        app = QApplication(sys.argv)
        gui = RobamineGUI()
        gui.show()
        sys.exit(app.exec_())

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--no-gui', default=False, action='store_true', help='Set to True if you want to use the command line interface')
    parser.add_argument('--yml', type=str, default='clutter', help='The yaml file to load')
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

if __name__ == '__main__':
    args = parse_args()
    run(**args)
