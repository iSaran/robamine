from random import Random
from collections import deque
import pickle
import numpy as np
from robamine.algo.util import Transition
import h5py

class ReplayBuffer:
    """
    Implementation of the replay experience buffer. Creates a buffer which uses
    the deque data structure. Here you can store experience transitions (i.e.: state,
    action, next state, reward) and sample mini-batches for training.

    You can  retrieve a transition like this:

    Example of use:

    .. code-block:: python

        replay_buffer = ReplayBuffer(10)
        replay_buffer.store()
        replay_buffer.store([0, 2, 1], [1, 2], -12.9, [2, 2, 1], 0)
        # ... more storing
        transition = replay_buffer(2)


    Parameters
    ----------
    buffer_size : int
        The buffer size
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0
        self.random = Random()

    def __call__(self, index):
        """
        Returns a transition from the buffer.

        Parameters
        ----------
        index : int
            The index number of the desired transition

        Returns
        -------
        tuple
            The transition

        """
        return self.buffer[index]

    def store(self, transition):
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
            1 if this state is terminal. 0 otherwise.
        """
        if self.count < self.buffer_size:
            self.count += 1
        else:
            self.buffer.popleft()
        self.buffer.append(transition)

    def sample_batch(self, given_batch_size):
        """
        Samples a minibatch from the buffer.

        Parameters
        ----------
        given_batch_size : int
            The size of the mini-batch.

        Returns
        -------
        numpy.array
            The state batch
        numpy.array
            The action batch
        numpy.array
            The reward batch
        numpy.array
            The next state batch
        numpy.array
            The terminal batch
        """
        batch = []

        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch = self.random.sample(self.buffer, batch_size)

        state_batch = np.array([_.state for _ in batch])
        action_batch = np.array([_.action for _ in batch])
        reward_batch = np.array([_.reward for _ in batch])
        next_state_batch = np.array([_.next_state for _ in batch])
        terminal_batch = np.array([_.terminal for _ in batch])

        return Transition(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)

    def clear(self):
        """
        Clears the buffer my removing all elements.
        """
        self.buffer.clear()
        self.count = 0

    def size(self):
        """
        Returns the current size of the buffer.

        Returns
        -------
        int
            The number of existing transitions.
        """
        return self.count

    def seed(self, random_seed):
        self.random.seed(random_seed)

    def save(self, file_path):
        b = {}
        b['buffer'] = self.buffer
        b['buffer_size'] = self.buffer_size
        b['count'] = self.count
        pickle.dump(b, open(file_path, 'wb'))

    @classmethod
    def load(cls, file_path):
        b = pickle.load(open(file_path, 'rb'))
        self = cls(b['buffer_size'])
        self.buffer = b['buffer']
        self.count = b['count']
        return self

    def merge(self, replay):
        if self.size() + replay.size() < self.buffer_size:
            self.buffer += replay.buffer
            self.count += replay.count
        else:
            raise RuntimeError('Buffer overflow during attempting merging.')

    def remove(self, index):
        del self.buffer[index]
        self.count -= 1


def get_batch_indices(dataset_size, batch_size, shuffle=True, seed=None):
    indices = np.arange(0, dataset_size, 1)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    total_size = len(indices)
    batch_size_ = min(batch_size, total_size)
    residual = total_size % batch_size_
    if residual > 0:
        for_splitting = indices[:-residual]
    else:
        for_splitting = indices
    batches = np.split(for_splitting, (total_size - residual) / batch_size_)
    return batches

class LargeReplayBuffer:
    def __init__(self, buffer_size, obs_dims, action_dim, path):
        self.buffer_size = buffer_size
        self.count = 0
        self.n_rows = 1
        self.obs_dims = obs_dims

        self.file = h5py.File(path, "a")

        for key in obs_dims:
            init_shape = obs_dims[key].copy()
            init_shape.insert(0, 1)
            init_shape = tuple(init_shape)
            max_shape = obs_dims[key].copy()
            max_shape.insert(0, buffer_size)
            max_shape = tuple(max_shape)
            self.file.create_dataset(key, init_shape, dtype='f', maxshape=max_shape)
            self.file.create_dataset('next_' + key, init_shape, dtype='f', maxshape=max_shape)
        self.action = self.file.create_dataset('action', (1, action_dim), dtype='f', maxshape=(buffer_size, action_dim))
        self.reward = self.file.create_dataset('reward', (1, 1), dtype='f', maxshape=(buffer_size, 1))
        self.terminal = self.file.create_dataset('terminal', (1, 1), dtype='i', maxshape=(buffer_size, 1))

        self.rng = Random()

    def __del__(self):
        self.file.close()

    def __call__(self, i):
        result = []
        for j in i:
            result.append(self._get_elem(j))
        return result

    def _get_elem(self, i):
        state, next_state = {}, {}
        for key in self.obs_dims:
            state[key] = self.file[key][i, :]
            next_state[key] = self.file['next_' + key][i, :]

        return Transition(state=state, next_state=next_state, action=self.action[i, :],
                          reward=self.reward[i, :], terminal=self.terminal[i, :])

    def resize(self, size):
        for key in self.obs_dims:
            self.file[key].resize(size, axis=0)
            self.file['next_' + key].resize(size, axis=0)
        self.action.resize(size, axis=0)
        self.reward.resize(size, axis=0)
        self.terminal.resize(size, axis=0)

    def store(self, transition):
        if self.count < self.buffer_size and self.n_rows < self.buffer_size and self.count > 0:
            self.n_rows += 1
            self.resize(self.n_rows)

        if self.count >= self.buffer_size:
            self.count = 0

        for key in self.obs_dims:
            self.file[key][self.count, :] = transition.state[key]
            self.file['next_' + key][self.count, :] = transition.next_state[key]

        self.action[self.count, :] = transition.action
        self.reward[self.count, :] = transition.reward
        self.terminal[self.count, :] = transition.terminal

        self.count += 1

    def clear(self):
        """
        Clears the buffer my removing all elements.
        """
        self.resize(1)
        self.n_rows = 1
        self.count = 0

    def size(self):
        return self.n_rows

    def seed(self, random_seed):
        self.rng.seed(random_seed)

    def save(self, file_path):
        raise NotImplementedError()

    @classmethod
    def load(cls, file_path):
        raise NotImplementedError()

    def merge(self, replay):
        raise NotImplementedError()

    def remove(self, index):
        raise NotImplementedError()

