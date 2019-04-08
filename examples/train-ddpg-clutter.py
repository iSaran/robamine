import tensorflow as tf
import logging
import robamine as rm


if __name__ == '__main__':
    rm.rb_logging.init('/tmp/robamine_logs/ddpg-clutter')
    logger = logging.getLogger('robamine')
    rm.seed_everything(999)

    world = rm.World(rm.DDPGParams(), 'Clutter-v1')
    world.train_and_eval(n_episodes_to_train=1000, n_episodes_to_evaluate=10, evaluate_every=25, save_every=10, print_progress_every=10, render_train=True)
