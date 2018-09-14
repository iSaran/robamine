import gym
from rlrl_py.algo.core import Agent
from rlrl_py.algo.ddpg.replay_buffer import ReplayBuffer
from rlrl_py.algo.ddpg.actor import Actor, TargetActor
#from rlrl_py.algo.ddpg.actor import Actor
import tensorflow as tf

class DDPG(Agent):
    def __init__(self, sess, env, random_seed, n_episodes, render,
            replay_buffer_size, actor_hidden_units, actor_final_layer_init,
            batch_size, actor_learning_rate, actor_tau):
        self.sess = sess
        Agent.__init__(self, env, random_seed, n_episodes, render)

        # Replay Buffer setup
        self.replay_buffer = ReplayBuffer(replay_buffer_size, random_seed)

        # Actor and its target network setup
        state_dim = int(self.env.observation_space.spaces['observation'].shape[0])
        action_dim = int(self.env.action_space.shape[0])
        self.actor = Actor(self.sess, state_dim, actor_hidden_units, action_dim, actor_final_layer_init, batch_size, actor_learning_rate)
        self.target_actor = TargetActor(self.actor, actor_tau)
        self.target_actor.update_params()

        self.sess.run(tf.global_variables_initializer())

    def do_exploration(self, state):
        obs = state['observation'].reshape(1, state['observation'].shape[0])
        return self.actor.predict(obs).squeeze()

    def learn(self):
        target_actor.update_params()

