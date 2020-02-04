import pickle
import os
import numpy as np
from robamine.algo.util import EnvData, Dataset

from robamine.algo.util import AutoEncoder
import torch.optim as optim

def load_dataset(path):
    env_data = EnvData.load(os.path.join(path, 'samples.env'))
    n_samples = len(env_data.transitions)
    s = env_data.transitions[0].state.shape
    array = np.zeros((n_samples, s[0], s[1], s[2]))
    for i in range(n_samples):
        array[i, :, :, :] = env_data.transitions[i].state
    return array

if __name__ == '__main__':
    data = load_dataset('/home/mkiatos/robamine/logs/robamine_logs_2020.02.03.18.36.16.854111/')

    # np array to Dataset in order to have split, minibatches functionalities
    dataset = Dataset.from_array(data, data)
    train_dataset, test_dataset = dataset.split(0.7)

    n_epochs = 150
    device = 'cpu'
    batch_size = 64
    learning_rate = 0.0001

    ae = AutoEncoder()

    optimizer = optim.Adam(ae.parameters(), lr=learning_rate)
    loss = nn.MSELoss()


    for epoch in range(n_epochs):
        # Calculate train loss
        train_x, train_y = train_dataset.to_array()
        real_x = torch.FloatTensor(train_x).to(device)
        prediction = ae(real_x)
        real_y = torch.FloatTensor(train_y).to(device)
        loss = loss(prediction, real_y)
        train_loss = loss.detach().cpu().numpy().copy()

        # Calculate loss in test dataset
        test_x, test_y = test_dataset.to_array()
        real_x = torch.FloatTensor(test_x).to(device)
        prediction = ae(real_x)
        real_y = torch.FloatTensor(test_y).to(device)
        loss = loss(prediction, real_y)
        test_loss = loss.detach().cpu().numpy().copy()

        print('train_loss:', train_loss, 'test_loss', test_loss)

        # Minimbatch update of network
        minibatches = train_dataset.to_minibatches(batch_size)
        for minibatch in minibatches:
            batch_x, batch_y = minibatch.to_array()

            real_x = torch.FloatTensor(batch_x).to(device)
            prediction = ae(real_x)
            real_y = torch.FloatTensor(batch_y).to(device)
            loss = loss(prediction, real_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # train with data
    # ...
