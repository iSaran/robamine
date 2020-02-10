
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

# STEPS = [1500, 1700]
STEPS = None

def state2numpy_array(data):
    steps = len(data)
    arrays = {}

    # Initiaze np arrays for plotting
    for key in data[0]:
        if key != 'time':
            arrays[key] = np.zeros((steps, data[0][key].shape[0]))

    time = np.zeros(steps)

    for i in range(steps):
        time[i] = data[i]['time']
        for key in data[i]:
            if key != 'time':
                arrays[key][i, :] = data[i][key]


    return arrays, time, steps


def run():

    with open('log.pkl', "rb") as fp:   # Unpickling
        data = pickle.load(fp)

    data, time, timesteps = state2numpy_array(data)


    if STEPS:
        start_time = STEPS[0]
        end_time = STEPS[1]
    else:
        start_time = 0
        end_time = timesteps

    fig, axs = plt.subplots(5, 3, sharex = True, sharey = False)
    axs = axs.ravel()
    plot_ind = 0    # Just an index to keep track of the plots

    plots = []
    for key in data:
        if data[key].shape[1] == 3:
            legend = ['x', 'y', 'z']
        elif data[key].shape[1] == 6:
            legend = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        else:
            legend = None
        plots.append({'signal': data[key], 'title': key, 'legend': legend})

    for i in range(len(plots)):
        axs[i].plot(time[start_time:end_time], plots[i]['signal'][start_time:end_time])
        axs[i].set_title(plots[i]['title'])
        if plots[i]['legend']:
            print('fdafdas')
            axs[i].legend(plots[i]['legend'])

    plt.show()

if __name__ == '__main__':
    run()
