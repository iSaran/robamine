from random import Random
from collections import deque
import pickle
import numpy as np
from robamine.algo.util import Transition
import h5py
from math import floor

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


class LargeReplayBuffer:
    def __init__(self, buffer_size, observation_dim, action_dim, path, existing=False):
        self.buffer_size = buffer_size
        self.count = 0
        self.n_rows = 1

        self.file = h5py.File(path, "a")

        if not existing:
            self.obs = self.file.create_dataset('obs', (1, 2, observation_dim, observation_dim), dtype='f',
                                                maxshape=(buffer_size, 2, observation_dim, observation_dim))
            self.next_obs = self.file.create_dataset('next_obs', (1, 2, observation_dim, observation_dim),
                                                     maxshape=(buffer_size, 2, observation_dim, observation_dim), dtype='f')
            self.action = self.file.create_dataset('action', (1, action_dim), dtype='f', maxshape=(buffer_size, action_dim))
            self.reward = self.file.create_dataset('reward', (1, 1), dtype='f', maxshape=(buffer_size, 1))
            self.terminal = self.file.create_dataset('terminal', (1, 1), dtype='i', maxshape=(buffer_size, 1))
        else:
            self.obs = self.file['obs']
            self.next_obs = self.file['next_obs']
            self.action = self.file['action']
            self.reward = self.file['reward']
            self.terminal = self.file['terminal']

        self.rng = Random()

    def __del__(self):
        self.file.close()

    def __call__(self, i):
        result = []
        for j in i:
            result.append(self._get_elem(j))
        return result

    def _get_elem(self, i):
        state = {
            'heightmap': self.obs[i, 0, :],
            'mask': self.obs[i, 1, :]
        }

        for key in list(self.obs.attrs.keys()):
            state[key] = self.obs.attrs[key]

        next_state = {
            'heightmap': self.next_obs[i, 0, :],
            'mask': self.next_obs[i, 1, :]
        }

        for key in list(self.next_obs.attrs.keys()):
            next_state[key] = self.next_obs.attrs[key]

        return Transition(state=state, next_state=next_state, action=self.action[i, :],
                          reward=self.reward[i, :], terminal=self.terminal[i, :])

    def resize(self, size):
        self.obs.resize(size, axis=0)
        self.next_obs.resize(size, axis=0)
        self.action.resize(size, axis=0)
        self.reward.resize(size, axis=0)
        self.terminal.resize(size, axis=0)

    def store(self, transition):

        if self.count < self.buffer_size and self.n_rows < self.buffer_size and self.count > 0:
            self.n_rows += 1
            self.resize(self.n_rows)

        if self.count >= self.buffer_size:
            self.count = 0

        self.obs[self.count, 0, :] = transition.state['heightmap']
        self.obs[self.count, 1, :] = transition.state['mask']
        for key in list(transition.state.keys()):
            if key != 'heightmap' and key != 'mask':
                self.obs.attrs[key] = transition.state[key]
        self.next_obs[self.count, 0, :] = transition.next_state['heightmap']
        self.next_obs[self.count, 1, :] = transition.next_state['mask']
        for key in list(transition.next_state.keys()):
            if key != 'heightmap' and key != 'mask':
                self.next_obs.attrs[key] = transition.next_state[key]
        self.action[self.count, :] = transition.action
        self.reward[self.count, :] = transition.reward
        self.terminal[self.count, :] = transition.terminal

        self.count += 1

    def sample(self, elements):
        indices = np.arange(0, self.n_rows, 1)
        self.rng.shuffle(indices)

        transitions = []
        for i in range(min(elements, self.size())):
            transitions.append(self(indices[i]))

        return transitions

    def sample_batch(self, given_batch_size):
        batch = self.sample(given_batch_size)

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
        with h5py.File(file_path, "r") as f:
            buffer_size = f['obs'].maxshape[0]
            observation_dim = f['obs'].shape[2]
            action_dim = f['action'].shape[1]
            n_rows = f['obs'].shape[0]
        buffer = cls(buffer_size=buffer_size, observation_dim=observation_dim, action_dim=action_dim, path=file_path,
                     existing=True)
        buffer.n_rows = n_rows
        buffer.count = n_rows - 1
        return buffer

    def merge(self, replay):
        raise NotImplementedError()

    def remove(self, index):
        raise NotImplementedError()


class RotatedLargeReplayBuffer(LargeReplayBuffer):
    def __init__(self, buffer_size, obs_shapes, action_dim, path, existing=False, rotations=1, mode="a"):
        self.buffer_size = buffer_size
        self.count = 0
        self.n_scenes = 1
        self.rotations = rotations
        self.action_dim = action_dim
        self.rotations = rotations

        self.file = h5py.File(path, mode)

        if not existing:
            self.file.create_group('obs')
            for key in list(obs_shapes.keys()):
                self.file['obs'].create_dataset(key, (self.n_scenes,) + obs_shapes[key], dtype='f',
                                         maxshape=(buffer_size,) + obs_shapes[key])

            self.file.create_dataset('action', (self.n_scenes, rotations, action_dim), dtype='f',
                                     maxshape=(buffer_size, rotations, action_dim))
            self.file.create_dataset('reward', (self.n_scenes, rotations, 1), dtype='f',
                                                   maxshape=(buffer_size, rotations, 1))

        self.rng = Random()

    def _get_elem(self, i):
        global_index = floor(i / self.rotations)
        local_index = i - global_index * self.rotations
        state = {}
        for key in list(self.file['obs'].keys()):
            state[key] = self.file['obs'][key][global_index, :]

        return Transition(state=state, next_state=None, action=self.file['action'][global_index, local_index, :],
                          reward=self.file['reward'][global_index, local_index, :], terminal=None)

    def get_transition(self, i):
        return self._get_elem(i)

    def get_scene(self, i):
        global_index = i
        state = {}
        for key in list(self.file['obs'].keys()):
            state[key] = self.file['obs'][key][global_index, :]

        return Transition(state=state, next_state=None, action=self.file['action'][global_index, :],
                          reward=self.file['reward'][global_index, :], terminal=None)


    def resize(self, size):
        for key in list(self.file['obs'].keys()):
            self.file['obs'][key].resize(size, axis=0)

        self.file['action'].resize(size, axis=0)
        self.file['reward'].resize(size, axis=0)


    def store(self, transition):

        if self.count < self.buffer_size and self.n_scenes < self.buffer_size and self.count > 0:
            self.n_scenes += 1
            self.resize(self.n_scenes)

        if self.count >= self.buffer_size:
            self.count = 0

        for key in list(self.file['obs'].keys()):
            self.file['obs'][key][self.count, :] = transition.state[key]

        self.file['action'][self.count, :] = transition.action
        self.file['reward'][self.count, :] = transition.reward

        self.count += 1

    def sample(self, elements):
        raise NotImplementedError()

    def sample_batch(self, given_batch_size):
        raise NotImplementedError()

    def size(self):
        return self.n_scenes * self.rotations

    @classmethod
    def load(cls, file_path, mode="a"):
        with h5py.File(file_path, "r") as f:
            buffer_size = f['action'].maxshape[0]
            n_scenes = f['action'].shape[0]
            rotations = f['action'].shape[1]
            action_dim = f['action'].shape[2]
        buffer = cls(buffer_size=buffer_size, obs_shapes=None, action_dim=action_dim, path=file_path,
                     existing=True, rotations=rotations, mode=mode)
        buffer.n_scenes = n_scenes
        buffer.count = n_scenes - 1
        return buffer

    def merge(self, buffer):
        assert buffer.n_scenes + self.n_scenes < self.buffer_size
        n_scenes_new = self.n_scenes + buffer.n_scenes - 1
        self.resize(n_scenes_new)

        for key in list(self.file['obs'].keys()):
            self.file['obs'][key][self.n_scenes - 1:, :] = buffer.file['obs'][key]
        self.file['action'][self.n_scenes - 1:, :] = buffer.file['action']
        self.file['reward'][self.n_scenes - 1:, :] = buffer.file['reward']

        self.n_scenes += buffer.n_scenes
        self.count += self.n_scenes

