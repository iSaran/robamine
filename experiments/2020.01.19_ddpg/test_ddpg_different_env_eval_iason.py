from robamine.algo.splitddpg import SplitDDPG
from robamine import rb_logging
import logging
import yaml

# For new world
import gym
import pickle
from robamine.algo.core import TrainEvalWorld, TrainingEpisode, TestingEpisode
import os

class TrainEvalWorldDifferentEnvs(TrainEvalWorld):
    def __init__(self, agent, env, env_eval, params, name=None):
        #n_episodes, render, save_every, eval_episodes, render_eval, eval_every
        super(TrainEvalWorldDifferentEnvs, self).__init__(agent, env, params, name)
        env_params = env_eval['params'] if 'params' in env_eval else {}
        self.env_eval = gym.make(env_eval['name'], params=env_params)

    def run_episode(self, i):
        episode = TrainingEpisode(self.agent, self.env)
        super(TrainEvalWorld, self).run_episode(episode, i)

        # Evaluate every some number of training episodes
        if (i + 1) % self.eval_every == 0:
            for j in range(self.eval_episodes):
                episode = TestingEpisode(self.agent, self.env_eval)
                episode.run(self.render_eval)
                self.eval_stats.update(self.eval_episodes * self.counter + j, episode.stats)
                self.episode_stats_eval.append(episode.stats)
                pickle.dump(self.episode_stats_eval, open(os.path.join(self.log_dir, 'episode_stats_eval.pkl'), 'wb'))

                for k in range (0, episode.stats['n_timesteps']):
                    self.expected_values_file.write(str(episode.stats['q_value'][k]) + ',' + str(episode.stats['monte_carlo_return'][k]) + '\n')
                    self.expected_values_file.flush()

                for k in range(len(episode.stats['actions_performed']) - 1):
                    self.actions_file.write(str(episode.stats['actions_performed'][k]) + ',')
                self.actions_file.write(str(episode.stats['actions_performed'][-1]) + '\n')
                self.actions_file.flush()

            self.counter += 1

def run():
    with open('params_iason.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    rb_logging.init(directory=params['world']['logging_dir'], friendly_name=params['world']['friendly_name'], file_level=logging.INFO)
    trainer = TrainEvalWorldDifferentEnvs(agent=params['agent'], env=params['env'], env_eval=params['env_eval'], params=params['world']['params'])
    trainer.run()

if __name__ == '__main__':
    run()
