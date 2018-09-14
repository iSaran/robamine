import gym

class Network:
    '''Abstract class which defines an interface for e Neural Network
    '''
    def __init__(self, sess, input_dim, hidden_dims, out_dims, name = ""):
        self.sess = sess
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dims
        self.name = name

        # Create online/learned network and the target network for the Actor
        self.inputs, self.out, self.net_params = self.create_architecture()

    def create_architecture(self):
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
    def __init__(self, env, random_seed, n_episodes, render):
        # Environment setup
        self.env = gym.make(env)
        self.env.seed(random_seed)
        self.episode_horizon = int(self.env._max_episode_steps)

        self.n_episodes = n_episodes
        self.render = render

    def train(self):
        for episode in range(self.n_episodes):
            state = self.env.reset()
            episode_reward = 0

            for t in range(self.episode_horizon):

                if (self.render):
                    self.env.render()

                action = self.do_exploration(state)
                next_state, reward, done, info = self.env.step(action)

                self.learn()

                episode_reward += reward
                state = next_state
            print('End of episode:', episode, 'with total reward: ', episode_reward)

    def do_exploration(self, state):
        return self.env.action_space.sample()

    def learn(self):
        pass

