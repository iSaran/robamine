from collections import deque
import random
import numpy as np

class ReplayBuffer:

    def __init__(self, buffer_size, seed=9876):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0
        random.seed(seed)

    def sample_batch(self, given_batch_size):
        batch = []

        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch = random.sample(self.buffer, self.count)
        state_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        reward_batch = np.array([_[2] for _ in batch])
        state_batch = np.array([_[3] for _ in batch])
        timestep_batch = np.array([_[4] for _ in batch])

        return state_batch, action_batch, reward_batch, state_batch, timestep_batch

    def store(self, state, action, reward, next_state, timestep):
        experience = (state, action, reward, next_state, timestep)
        if self.count < self.buffer_size:
            self.count += 1
        else:
            self.buffer.popleft()
        self.buffer.append(experience)

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def size(self):
        return self.count
