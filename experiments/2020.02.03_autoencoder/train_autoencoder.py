import pickle
import os
import numpy as np
from robamine.algo.util import EnvData, Dataset

from robamine.utils.cv_tools import AutoEncoder, AeLoss, plot_height_map
import torch
import torch.optim as optim
import torch.nn as nn

import cv2

from sklearn.decomposition import PCA


def compute_loss(train_dataset, test_dataset, ae, ae_loss):
    # Calculate train loss

    return train_loss, test_loss


def load_dataset(path):
    env_data = EnvData.load(os.path.join(path, 'samples.env'))
    n_samples = len(env_data.transitions)
    s = env_data.transitions[0].state.shape
    array = np.zeros((n_samples, s[0], s[1], s[2]))
    for i in range(n_samples):
        array[i, :, :, :] = env_data.transitions[i].state

    # pca_array = np.reshape(array, (n_samples, -1))
    # print(pca_array.shape)
    # pca = PCA(.95)
    # pca_components = pca.fit_transform(pca_array)
    # print('pca components:', pca_components.shape)
    # input('')

    return array

if __name__ == '__main__':
    torch.manual_seed(0)

    data = load_dataset('/home/mkiatos/robamine/logs/robamine_logs_2020.02.03.18.36.16.854111/')

    # np array to Dataset in order to have split, minibatches functionalities
    dataset = Dataset.from_array(data, data)
    train_dataset, test_dataset = dataset.split(0.8)
    train_dataset.seed(0)

    n_epochs = 50
    device = 'cuda'
    batch_size = 64
    learning_rate = 0.0001

    input_dim = [128, 128, 1]
    latent_dim = 500
    ae = AutoEncoder(input_dim, latent_dim).to(device)

    optimizer = optim.Adam(ae.parameters(), lr=learning_rate)
    # ae_loss = nn.MSELoss()
    # ae_loss = nn.BCELoss()
    ae_loss = AeLoss()
    # ToDo: loss is the sum of regression and classification

    for epoch in range(n_epochs):
        train_loss = 0.0
        test_loss = 0.0
        minib = train_dataset.to_minibatches(batch_size)
        for minibatch in minib:
            batch_x, batch_y = minibatch.to_array()
            real_x = torch.tensor(np.expand_dims(batch_x[:, 0, :, :], axis=1), dtype=torch.float, requires_grad=True).to(device)
            real_y_1 = torch.tensor(batch_y[:, 0, :, :], dtype=torch.float, requires_grad=True).to(device)
            real_y_2 = torch.tensor(batch_y[:, 1, :, :], dtype=torch.float, requires_grad=True).to(device)
            pred_1, pred_2 = ae(real_x)
            pred_1 = torch.squeeze(pred_1)
            pred_2 = torch.squeeze(pred_2)
            loss = ae_loss(real_y_1, pred_1, real_y_2, pred_2)
            train_loss += loss.detach().cpu().numpy().copy()

        train_loss /= len(minib)

        # # # Calculate loss in test dataset
        # test_x, test_y = test_dataset.to_array()
        # n_samples = test_x.shape[0]
        # for i in range(n_samples):
        #     real_x = torch.tensor(np.expand_dims(test_x[i], axis=0), dtype=torch.float, requires_grad=True).to(device)
        #     pred = ae(real_x)
        #     loss = ae_loss(real_x, pred)
        #     test_loss += loss.detach().cpu().numpy().copy()
        # test_loss /= n_samples

        print('epoch:', epoch, 'train_loss:', train_loss, 'test_loss', test_loss)

        # # Minimbatch update of network
        minibatches = train_dataset.to_minibatches(batch_size)
        for minibatch in minibatches:
            batch_x, batch_y = minibatch.to_array()
            real_x = torch.tensor(np.expand_dims(batch_x[:, 0, :, :], axis=1), dtype=torch.float, requires_grad=True).to(device)
            real_y_1 = torch.tensor(batch_y[:, 0, :, :], dtype=torch.float, requires_grad=True).to(device)
            real_y_2 = torch.tensor(batch_y[:, 1, :, :], dtype=torch.float, requires_grad=True).to(device)
            pred_1, pred_2 = ae(real_x)
            pred_1 = torch.squeeze(pred_1)
            pred_2 = torch.squeeze(pred_2)
            loss = ae_loss(real_y_1, pred_1, real_y_2, pred_2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_x, test_y = train_dataset.to_array()
    for i in range(10):
        x = test_x[i, :, :, :]
        real_x = torch.tensor(np.expand_dims(np.expand_dims(x[0, :, :], axis=0), axis=0), dtype=torch.float, requires_grad=True).to(device)
        pred_1, pred_2 = ae(real_x)
        pred_1 = torch.squeeze(pred_1)
        pred_1 = pred_1.detach().cpu().numpy()

        pred_2 = torch.squeeze(pred_2)
        mask_x = torch.ones(128, 128).to(device)
        mask_y = torch.zeros(128, 128).to(device)
        pred_2 = torch.where(pred_2 > 0.5, mask_x, mask_y)
        pred_2 = pred_2.detach().cpu().numpy()

        plot_height_map(x[0, :, :])
        plot_height_map(pred_1)
        plot_height_map(x[1, :, :])
        plot_height_map(pred_2)

