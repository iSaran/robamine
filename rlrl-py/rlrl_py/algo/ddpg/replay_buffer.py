from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, seed=9876):
        """
        Creates a buffer using the deque data structure.

        Parameters
        ----------
        buffer_size : int
            The buffer size
        seed : int
            A seed for initializing the random batch sampling
        """
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0
        random.seed(seed)

    def store(self, state, action, reward, next_state, terminal):
        """
        Stores a new transition on the buffer.

        Parameters
        ----------
        state : np.ndarray
            The state of the transition
        action : np.ndarray
            The action of the transition
        reward : np.float32
            The reward of the transition
        next_state : np.ndarray
            The next state of the transition
        terminal : np.float32
            If this state is terminal. 0 if it is not, 1 if it is.
        """
        transition = (state, action, reward, next_state, terminal)
        if self.count < self.buffer_size:
            self.count += 1
        else:
            self.buffer.popleft()
        self.buffer.append(transition)

    def sample_batch(self, given_batch_size):
        batch = []

        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch = random.sample(self.buffer, batch_size)

        state_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        reward_batch = np.array([_[2] for _ in batch])
        next_state_batch = np.array([_[3] for _ in batch])
        terminal_batch = np.array([_[4] for _ in batch])

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def size(self):
        return self.count
