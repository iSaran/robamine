from replay_buffer import ReplayBuffer
from actor import Actor, TargetActor
from critic import Critic, TargetCritic
import yaml
import tensorflow as tf
import tflearn


if __name__ == '__main__':
    #replay_buffer = ReplayBuffer(100)
    #print('Size: ', replay_buffer.size())
    #replay_buffer.add(0, 0, -1, 1, 0)
    #replay_buffer.add(1, 0, -1, 3, 0)
    #replay_buffer.add(4, 2, -1, 1, 0)
    #replay_buffer.add(7, 2, -1, 2, 0)
    #replay_buffer.add(0, 1, -1, 8, 0)
    #print('Size: ', replay_buffer.size())
    #print('Buffer: ', replay_buffer.buffer)
    #batch = replay_buffer.sample_batch(2)
    #print(batch)


    yml_file = open("params.yml", 'r')
    params = yaml.load(yml_file)

    with tf.Session() as sess:
        #actor = Actor(sess, state_dim=10, action_dim=2, n_units=params['actor']['n_units'], final_layer_init=params['actor']['final_layer_init'], tau=params['actor']['tau'], batch_size=params['actor']['tau'], learning_rate = 11)

        actor = Actor(sess, 10, [10, 10], 20, [-2, 2], 64, 0.001)
        target_actor = TargetActor(actor, 0.001)
        critic = Critic(sess, (10, 5), [10, 10], 20, [-2, 2], 64, 0.001)
        target_critic = TargetCritic(critic, 0.001)
        #writer = tf.summary.FileWriter("/tmp/basic", sess.graph)
