#!/usr/bin/env python3

import matplotlib.pyplot as plt
import csv
import os

reference_Q_mean = []
reference_action_mean = []
reference_perturbed_action_mean = []
reference_actor_Q_std = []
param_noise_stddev = []
reference_Q_std = []
reference_perturbed_action_std = []
total_epochs = []
rollout_episode_steps = []
rollout_Q_mean = []
reference_actor_Q_mean = []
total_episodes = []
rollout_actions_mean = []
rollout_episodes = []
total_duration = []
train_loss_actor = []
total_steps_per_second = []
obs_rms_std = []
train_loss_critic = []
rollout_actions_std = []
rollout_return_history = []
reference_action_std = []
rollout_return = []
train_param_noise_distance = []
obs_rms_mean = []
total_steps = []


directory = '/tmp/openai-2018-07-10-15-18-28-893568'
with open(os.path.join(directory, 'progress.csv'), 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
       reference_Q_mean = row["reference_Q_mean"]
       reference_action_mean = row["reference_action_mean"]
       reference_perturbed_action_mean = row["reference_perturbed_action_mean"]
       reference_actor_Q_std = row["reference_actor_Q_std"]
       param_noise_stddev = row["param_noise_stddev"]
       reference_Q_std = row["reference_Q_std"]
       reference_perturbed_action_std = row["reference_perturbed_action_std"]
       total_epochs = row["total/epochs"]
       rollout_episode_steps = row["rollout/episode_steps"]
       rollout_Q_mean = row["rollout/Q_mean"]
       reference_actor_Q_mean = row["reference_actor_Q_mean"]
       total_episodes = row["total/episodes"]
       rollout_actions_mean = row["rollout/actions_mean"]
       rollout_episodes = row["rollout/episodes"]
       total_duration = row["total/duration"]
       train_loss_actor = row["train/loss_actor"]
       total_steps_per_second = row["total/steps_per_second"]
       obs_rms_std = row["obs_rms_std"]
       train_loss_critic = row["train/loss_critic"]
       rollout_actions_std = row["rollout/actions_std"]
       rollout_return_history = row["rollout/return_history"]
       reference_action_std = row["reference_action_std"]
       rollout_return = row["rollout/return"]
       train_param_noise_distance = row["train/param_noise_distance"]
       obs_rms_mean = row["obs_rms_mean"]
       total_steps = row["total/steps"]
       
       print(total_epochs)

plt.plot(total_epochs, rollout_return)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.show()
