import gym
from rlrl_py.algo.core import Agent
from rlrl_py.algo.ddpg.replay_buffer import ReplayBuffer
from rlrl_py.algo.ddpg.actor_critic import Actor, TargetActor, Critic, TargetCritic
from rlrl_py.algo.ddpg.ou_noise import OrnsteinUhlenbeckActionNoise
#from rlrl_py.algo.ddpg.actor import Actor
import tensorflow as tf
import numpy as np

class DDPG(Agent):
    def __init__(self, sess, env, random_seed=10000000, log_dir='/tmp',
            replay_buffer_size=1e6, actor_hidden_units=(400, 300), final_layer_init=(-3e-3, 3e-3),
            batch_size=64, actor_learning_rate=1e-4, tau=1e-3, critic_hidden_units=(400, 300),
            critic_learning_rate=1e-3, gamma=0.999, exploration_noise_sigma=0.1):
        self.sess = sess
        self.gamma = gamma
        Agent.__init__(self, sess, env, random_seed, log_dir, name="DDPG")

        # Initialize the Actor network and its target net
        state_dim = int(self.env.observation_space.shape[0])
        action_dim = int(self.env.action_space.shape[0])
        self.actor = Actor(self.sess, state_dim, actor_hidden_units, action_dim, final_layer_init, batch_size, actor_learning_rate)
        self.target_actor = TargetActor(self.actor, tau)

        # Initialize the Critic network and its target net
        self.critic = Critic(self.sess, (state_dim, action_dim), critic_hidden_units, final_layer_init, critic_learning_rate)
        self.target_critic = TargetCritic(self.critic, tau)

        # Initialize target networks with weights equal to the learned networks
        self.sess.run(tf.global_variables_initializer())
        self.target_actor.equalize_params()
        self.target_critic.equalize_params()

        # Initialize replay buffer
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(self.batch_size, random_seed)

        self.exploration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma = exploration_noise_sigma)

    def explore(self, state):
        obs = state.reshape(1, state.shape[0])
        return self.actor.predict(obs).squeeze() + self.exploration_noise()

    def learn(self, state, action, reward, next_state, done):
        # Store the transition into the replay buffer
        self.replay_buffer.store(state, action, reward, next_state, done)

        # If we have not enough samples just keep storing transitions to the
        # buffer and thus exit.
        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample a mini batch from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.replay_buffer.sample_batch(self.batch_size)

        # Calculate the target of the Q value from the target network
        next_action_batch = self.target_actor.predict(next_state_batch)
        target_q = self.target_critic.predict((next_state_batch, next_action_batch))
        y_i = []
        for k in range(self.batch_size):
            if terminal_batch[k]:
                y_i.append(reward_batch[k])
            else:
                y_i.append(reward_batch[k] + self.gamma * target_q[k])

        ##############3# Update Critic by minimizing the loss

        ##############3self.target_actor.update_params()

