import gym
import tensorflow as tf
import rlrl_py.algo.util as util

class Network:
    '''Base class which defines an interface for a Neural Network
    '''
    def __init__(self, sess, input_dim, hidden_dims, out_dim, name = ""):
        self.sess = sess
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.name = name

        # Create online/learned network and the target network for the Actor
        self.inputs, self.out, self.net_params = self.create_architecture()

    def create_architecture(self):
        ''' Creates the architecture of the NN. It should be implemented by any
        class which inherits.
        '''
        raise NotImplementedError

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={self.inputs: inputs})

    def get_params(self, name = None):
        if name is None:
            return self.sess.run(self.net_params)
        else:
            k = [v for v in self.net_params if v.name == name][0]
            return self.sess.run(k)

class Agent:
    '''Base class which defines an RL agent
    '''
    def __init__(self, sess, env, random_seed=10000000, log_dir='/tmp', name=None):
        # Environment setup
        self.env = gym.make(env)
        self.env = gym.wrappers.FlattenDictWrapper(self.env, ['observation', 'desired_goal'])
        self.env.seed(random_seed)
        self.episode_horizon = int(self.env.env._max_episode_steps)

        self.sess = sess
        self.name = name

        self.logger = util.Logger(sess, log_dir, self.name, env)

    def train(self, n_episodes, render=True):
        """
        Provides the training or optimization loop. For a given number of
        episodes this function runs the basic RL loop for each timestep of the
        episode:
            - Selects an action based on the exploration policy
            - Performs the action to the environment and reads the next state
              and the reward.
            - Learns from this experience (i.e. optimizes the learned policy
              based on some algorithm).
        self.predict method) and provides stats.

        Parameters
        ----------
        n_episodes:
            The number of episodes to train for.
        render:
            True if rendering is desired during training.
        """
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0

            for t in range(self.episode_horizon):

                if (render):
                    self.env.render()

                # Select an action based on the exploration policy
                action = self.explore(state)

                # Execute the action on the environment  and observe reward and next state
                next_state, reward, done, info = self.env.step(action)

                # Learn
                self.learn(state, action, reward, next_state, done)

                episode_reward += reward
                state = next_state

                if done:
                    break

            self.logger.log(episode_reward, episode)
            self.logger.print_console(episode, n_episodes)

    def learn(self, state, action, reward, next_state, done):
        """
        The function which implements the algorithm for learning from the
        experience (i.e. optimize the learn policy). Used by the loop in
        self.train()
        """
        pass

    def explore(self, state):
        """
        Provides explorative actions. In the general case the policy for
        exploration may be different from the actual policy that we want to
        optimize (and thus the explore and the evaluate functions may be
        different).

        Parameters
        ----------

        state:
            The current state of the environment.

        Returns
        -------
        An action to be performed for exploration.
        """
        return self.env.action_space.sample()

    def evaluate(self, n_episodes, render=True):
        """
        Provides the evaluation loop. For a given number of episodes this
        function runs the optimal policy that the agent has learned (see
        self.predict method) and provides stats.

        Parameters
        ----------
        n_episodes:
            The number of episodes to evaluate for.
        render:
            True if rendering is desired during evaluation.
        """
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0

            for t in range(self.episode_horizon):

                if (render):
                    self.env.render()

                # Select an action based on the exploration policy
                action = self.predict(state)

                # Execute the action on the environment  and observe reward and next state
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                state = next_state

                if done:
                    break

            self.logger.log(episode_reward, episode)
            self.logger.print_console(episode, n_episodes)

    def predict(self, state):
        """
        Provides the optimal actions based on the learned policy. It is the
        actual policy.

        Parameters
        ----------

        state:
            The current state of the environment.

        Returns
        -------
        An optimal action.
        """
        pass
