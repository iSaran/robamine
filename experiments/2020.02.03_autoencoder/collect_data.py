import gym
import yaml
import robamine
import logging
from robamine import rb_logging
from robamine.algo.core import SampleTransitionsWorld

#
# class HeightmapCollectorEpisode(Episode):
#     def __init__(self, agent, env):
#         super().__init__(agent, env)
#         self.heightmaps = []
#
#     def _action_policy(self, state):
#         return self.agent.predict(state)
#
#     def _learn(self, transition):
#         self.heightmaps.append(self.transition.state)
#
#
# class HeightmapCollector(RLWorld):
#     def __init__(self, agent, env, params, name=''):
#         super(HeightmapCollector, self).__init__(agent, env, params, name)
#         self.heightmaps = np.zeros((1, self.state_dim))
#
#     def run_episode(self, i):
#         episode = HeightmapCollectorEpisode(self.agent, self.env)
#         super().run_episode(episode, i)
#         for h in episode.heightmaps:
#             self.heightmaps = np.concatenate(self.heightmaps, h, axis=0)
#
#         with open(os.path.join(self.log_dir, 'data'), "w+") as file:
#             pickle.dump(self.heightmaps, file)

def run():
    with open('collect_data.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    rb_logging.init(directory=params['world']['logging_dir'], friendly_name=params['world']['friendly_name'], file_level=logging.INFO)
    SampleTransitionsWorld(params['agent'], params['env'], params['world']['params'], '').run()

if __name__ == '__main__':
    run()
