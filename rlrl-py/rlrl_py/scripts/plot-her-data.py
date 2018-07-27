#!/usr/bin/env python3

import matplotlib.pyplot as plt
import csv
import os
import numpy as np

import experiment_command as exps

# set_of_experiments = 'her'
#experiments = ['FetchSlide-v1', 'FetchPickAndPlace-v1', 'FetchReach-v1', 'FetchPush-v1', 'HandReach-v0', 'HandManipulateBlockRotateZ-v0', 'HandManipulateBlockRotateParallel-v0', 'HandManipulateBlockRotateXYZ-v0', 'HandManipulateBlockFull-v0', 'HandManipulatePenFull-v0']

# set_of_experiments = 'bhand-her'
#
database = exps.get_yml("./", "experiments")
_, experiments, _ = exps.get_exp_names_commands(database)

# The variable names existing in the progress.csv file in the directory of the experiment
y_variables = ["stats_g/std", "test/mean_Q", "stats_o/mean", "test/episode", "stats_o/std", "train/success_rate", "stats_g/mean", "train/episode", "test/success_rate"]
x_variable_name = "epoch"

scale = 1

for exp in experiments:
    average_across_epochs = range(0, database[exp]["params"]["n_epochs"])
    # Setup directories properly
    print('Processing experiment: ' + exp)
    directory = database[exp]["dir"] + "/" + exp
    if not os.path.exists(directory + '/plots'):
        os.makedirs(directory + '/plots')

    # Read the x_variable for this experiment
    x_variable = []
    with open(os.path.join(directory, 'progress.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x_variable.append(float(row[x_variable_name]))

    for v_name in y_variables:
        y_variable = []
        with open(os.path.join(directory, 'progress.csv'), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                y_variable.append(float(row[v_name]))

            y_variable = np.array(y_variable).reshape(len(average_across_epochs), int(len(x_variable)/len(average_across_epochs)))
            mean = np.mean(y_variable, axis=1)
            std = np.std(y_variable, axis=1)
            plt.plot(average_across_epochs, mean, color="#00b8e6", linewidth=2, label="Cross-validation score")
            plt.fill_between(average_across_epochs, mean - std, mean + std, color="#ccf5ff")
            plt.plot(average_across_epochs, y_variable, color="#33adff", label="Cross-validation score")

            #plt.plot(x_variable, y_variable)
            plt.xlabel('Epoch')
            plt.ylabel(v_name)
            plt.grid(color='#a6a6a6', linestyle='--', linewidth=0.5 )
            plt.savefig(directory + '/plots/'+v_name.replace('/', '_') +'.png')
            plt.clf()

