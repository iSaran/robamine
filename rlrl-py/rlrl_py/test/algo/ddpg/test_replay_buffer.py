import unittest
from rlrl_py.algo.ddpg.replay_buffer import ReplayBuffer
import numpy as np

class TestReplayBuffer(unittest.TestCase):
    def test_creating_object(self):
        replay_buffer = ReplayBuffer(10)

    def test_store(self):
        transitions = []
        transitions.append({'state': [1, 0, 0, 0], 'action': [1, 4], 'reward': -0.30, 'next_state': [1, 40, 3, 4], 'terminal': 0.0 })
        transitions.append({'state': [2, 2, 0, 0], 'action': [2, 3], 'reward': -10.3, 'next_state': [2, 40, 3, 4], 'terminal': 0.0 })
        transitions.append({'state': [3, 8, 1, 0], 'action': [3, 2], 'reward': -20.3, 'next_state': [3, 40, 3, 4], 'terminal': 1.0 })
        transitions.append({'state': [4, 3, 1, 1], 'action': [4, 1], 'reward': -30.3, 'next_state': [4, 40, 3, 4], 'terminal': 0.0 })

        replay_buffer = ReplayBuffer(10)

        for t in transitions:
            replay_buffer.store(t['state'], t['action'], t['reward'], t['next_state'], t['terminal'])

        self.assertEqual(replay_buffer.size(), 4)
        self.assertEqual(replay_buffer.buffer[2][0], [3, 8, 1, 0])

    def test_sample(self):
        transitions = []
        transitions.append({'state': [1, 0, 0, 0], 'action': [1, 4], 'reward': -0.30, 'next_state': [1, 40, 3, 4], 'terminal': 0.0 })
        transitions.append({'state': [2, 2, 0, 0], 'action': [2, 3], 'reward': -10.3, 'next_state': [2, 40, 3, 4], 'terminal': 0.0 })
        transitions.append({'state': [3, 8, 1, 0], 'action': [3, 2], 'reward': -20.3, 'next_state': [3, 40, 3, 4], 'terminal': 1.0 })
        transitions.append({'state': [4, 3, 1, 1], 'action': [4, 1], 'reward': -30.3, 'next_state': [4, 40, 3, 4], 'terminal': 0.0 })

        replay_buffer = ReplayBuffer(10, 1)

        for t in transitions:
            replay_buffer.store(t['state'], t['action'], t['reward'], t['next_state'], t['terminal'])

        state_batch, action_batch, reward_batch, next_state_batch, terminal = replay_buffer.sample_batch(2)

        self.assertTrue((state_batch[0] == np.array([2, 2, 0, 0])).all())
        self.assertTrue((state_batch[1] == np.array([3, 8, 1, 0])).all())
        self.assertTrue((action_batch[0] == np.array([2, 3])).all())
        self.assertTrue((action_batch[1] == np.array([3, 2])).all())
        self.assertEqual(reward_batch[0], -10.3)
        self.assertEqual(reward_batch[1], -20.3)
        self.assertTrue((next_state_batch[0] == np.array([2, 40, 3, 4])).all())
        self.assertTrue((next_state_batch[1] == np.array([3, 40, 3, 4])).all())
        self.assertEqual(terminal[0], 0.0)
        self.assertEqual(terminal[1], 1.0)
if __name__ == '__main__':
    unittest.main()
