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


directory = '/tmp/openai-2018-07-10-15-46-29-849140'
with open(os.path.join(directory, 'progress.csv'), 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
       reference_Q_mean.append(float(row["reference_Q_mean"]))
       reference_action_mean.append(float(row["reference_action_mean"]))
       reference_perturbed_action_mean.append(float(row["reference_perturbed_action_mean"]))
       reference_actor_Q_std.append(float(row["reference_actor_Q_std"]))
       param_noise_stddev.append(float(row["param_noise_stddev"]))
       reference_Q_std.append(float(row["reference_Q_std"]))
       reference_perturbed_action_std.append(float(row["reference_perturbed_action_std"]))
       total_epochs.append(float(row["total/epochs"]))
       rollout_episode_steps.append(float(row["rollout/episode_steps"]))
       rollout_Q_mean.append(float(row["rollout/Q_mean"]))
       reference_actor_Q_mean.append(float(row["reference_actor_Q_mean"]))
       total_episodes.append(float(row["total/episodes"]))
       rollout_actions_mean.append(float(row["rollout/actions_mean"]))
       rollout_episodes.append(float(row["rollout/episodes"]))
       total_duration.append(float(row["total/duration"]))
       train_loss_actor.append(float(row["train/loss_actor"]))
       total_steps_per_second.append(float(row["total/steps_per_second"]))
       obs_rms_std.append(float(row["obs_rms_std"]))
       train_loss_critic.append(float(row["train/loss_critic"]))
       rollout_actions_std.append(float(row["rollout/actions_std"]))
       rollout_return_history.append(float(row["rollout/return_history"]))
       reference_action_std.append(float(row["reference_action_std"]))
       rollout_return.append(float(row["rollout/return"]))
       train_param_noise_distance.append(float(row["train/param_noise_distance"]))
       obs_rms_mean.append(float(row["obs_rms_mean"]))
       total_steps.append(float(row["total/steps"]))

rewards = []
length = []
time = []
with open(os.path.join(directory, '0.monitor.csv'), 'r') as csvfile1:
    reader = csv.DictReader(csvfile1)
    for row in reader:
       length.append(float(row["l"]))
       time.append(float(row["t"]))
       rewards.append(float(row["r"]))


print(rollout_return)

plt.plot(total_steps, rollout_return)
plt.xlabel('Total Steps')
plt.ylabel('Rollout Return')
plt.show()

plt.plot(time, rewards)
plt.xlabel('Time')
plt.ylabel('Rewards')
plt.show()
