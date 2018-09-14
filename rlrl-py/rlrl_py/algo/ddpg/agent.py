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

        # Initialize the Actor network and its target net
        state_dim = int(self.env.observation_space.spaces['observation'].shape[0])
        action_dim = int(self.env.action_space.shape[0])
        self.actor = Actor(self.sess, state_dim, actor_hidden_units, action_dim, actor_final_layer_init, batch_size, actor_learning_rate)
        self.target_actor = TargetActor(self.actor, actor_tau)

        # Initialize the Critic network and its target net
        # ...

        self.sess.run(tf.global_variables_initializer())
        self.target_actor.update_params()

        # Initialize replay buffer
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(self.batch_size, random_seed)

    def do_exploration(self, state):
        obs = state['observation'].reshape(1, state['observation'].shape[0])
        return self.actor.predict(obs).squeeze()

    def learn(self, state, action, reward, next_state, done):

        # Store the transition into the replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # If we have not enough samples just keep storing transitions to the
        # buffer and thus exit.
        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample a mini batch from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.replay_buffer.sample_batch(self.batch_size)
