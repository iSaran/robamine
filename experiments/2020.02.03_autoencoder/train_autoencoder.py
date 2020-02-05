import pickle
import os
import numpy as np
from robamine.algo.util import EnvData, Dataset

from robamine.utils.cv_tools import AutoEncoder, AeLoss
import torch
import torch.optim as optim
import torch.nn as nn


def compute_loss(train_dataset, test_dataset, network):
    # Calculate train loss
    train_loss = 0.0
    test_loss = 0.0

    train_x, train_y = train_dataset.to_array()
    n_samples = train_x.shape[0]
    for i in range(n_samples):
        real_x = torch.tensor(np.expand_dims(train_x[i], axis=0), dtype=torch.float, requires_grad=True).to(device)
        pred_heigthmap, pred_mask = ae(real_x)
        pred_1 = torch.tensor(pred_heigthmap, dtype=torch.float, requires_grad=True).to(device)
        pred_2 = torch.tensor(pred_mask, dtype=torch.float, requires_grad=True).to(device)
        loss = ae_loss(real_x, pred_1, pred_2)
        train_loss += loss.detach().cpu().numpy().copy()
    train_loss /= n_samples

    # # Calculate loss in test dataset
    test_x, test_y = test_dataset.to_array()
    n_samples = test_x.shape[0]
    for i in range(n_samples):
        real_x = torch.tensor(np.expand_dims(test_x[i], axis=0), dtype=torch.float, requires_grad=True).to(device)
        pred_heigthmap, pred_mask = ae(real_x)
        pred_1 = torch.tensor(pred_heigthmap, dtype=torch.float, requires_grad=True).to(device)
        pred_2 = torch.tensor(pred_mask, dtype=torch.float, requires_grad=True).to(device)
        loss = ae_loss(real_x, pred_1, pred_2)
        test_loss += loss.detach().cpu().numpy().copy()

    return train_loss, test_loss


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
    device = 'cuda'
    batch_size = 64
    learning_rate = 0.0001

    input_dim = [128, 128, 2]
    latent_dim = 500
    ae = AutoEncoder(input_dim, latent_dim).to(device)

    optimizer = optim.Adam(ae.parameters(), lr=learning_rate)
    # loss = nn.MSELoss()
    ae_loss = AeLoss()
    # ToDo: loss is the sum of regression and classification

    for epoch in range(n_epochs):
        train_loss, test_loss = compute_loss(train_dataset, test_dataset, ae)
        print('epoch:', epoch, 'train_loss:', train_loss, 'test_loss', test_loss)

        # # Minimbatch update of network
        minibatches = train_dataset.to_minibatches(batch_size)
        for minibatch in minibatches:
            batch_x, batch_y = minibatch.to_array()

            real_x = torch.FloatTensor(batch_x).to(device)
            real_x = torch.tensor(batch_x, dtype=torch.float, requires_grad=True).to(device)
            pred_heigthmap, pred_mask = ae(real_x)
            pred_1 = torch.tensor(pred_heigthmap, dtype=torch.float, requires_grad=True).to(device)
            pred_2 = torch.tensor(pred_mask, dtype=torch.float, requires_grad=True).to(device)
            loss = ae_loss(real_x, pred_1, pred_2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
