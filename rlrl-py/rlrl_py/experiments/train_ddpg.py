import yaml
from rlrl_py.algo.ddpg.agent import DDPG
import tensorflow as tf

if __name__ == '__main__':

    yml_file = open("./config/ddpg.yml", 'r')
    params = yaml.load(yml_file)

    with tf.Session() as sess:
        agent = DDPG(sess, params['env'], params['random_seed'],
                params['n_episodes'], params['render'],
                params['replay_buffer_size'], params['actor']['hidden_units'],
                params['actor']['final_layer_init'], params['batch_size'],
                params['actor']['learning_rate'], params['tau'],
                params['critic']['hidden_units'],
                params['critic']['final_layer_init'],
                params['critic']['learning_rate'])

        agent.train()

