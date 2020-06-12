import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import h5py
import os
import pickle

from robamine.utils.cv_tools import Feature
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

LATENT_DIM = 256  # hardcoded

params = {
    'layers': 4,
    'encoder': {
        'filters': [16, 32, 64, 128],
        'kernels': [4, 4, 4, 4],
        'strides': [2, 2, 2, 2],
        'padding': [1, 1, 1, 1],
        'pool': [2, 2, 2, 2]
    },
    'decoder': {
        'filters': [128, 64, 32, 16],
        'kernels': [4, 4, 4, 4],
        'stride': [2, 2, 2, 2],
        'padding': [1, 1 ,1 , 1]
    },
    'latent_dim': LATENT_DIM,
    'device': 'cuda',
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 200
}

class Encoder(nn.Module):
    def __init__(self, latent_dim, params = params):
        super(Encoder, self).__init__()

        self.filters = params['encoder']['filters']
        self.kernels = params['encoder']['kernels']
        self.stride = params['encoder']['strides']
        self.pad = params['encoder']['padding']
        self.pool = params['encoder']['pool']
        self.no_of_layers = params['layers']

        self.conv1 = nn.Conv2d(1, self.filters[0], self.kernels[0], stride=self.stride[0], padding=self.pad[0])
        self.conv2 = nn.Conv2d(self.filters[0], self.filters[1], self.kernels[1], stride=self.stride[1], padding=self.pad[1])
        self.conv3 = nn.Conv2d(self.filters[1], self.filters[2], self.kernels[2], stride=self.stride[2], padding=self.pad[2])
        self.conv4 = nn.Conv2d(self.filters[2], self.filters[3], self.kernels[3], stride=self.stride[3], padding=self.pad[3])

        self.mu = nn.Linear(8192, latent_dim)
        self.logvar = nn.Linear(8192, latent_dim)
        self.fc = nn.Linear(8192, latent_dim)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = x.view(-1, 8192)
        # return self.mu(x), self.logvar(x)
        return nn.functional.relu(self.fc(x))

class Decoder(nn.Module):
    def __init__(self, latent_dim, params = params):
        super(Decoder, self).__init__()

        self.filters = params['decoder']['filters']
        self.kernels = params['decoder']['kernels']
        self.stride = params['decoder']['stride']
        self.pad = params['decoder']['padding']
        self.no_of_layers = params['layers']
        self.device = params['device']

        self.latent = nn.Linear(latent_dim, 8192)

        self.deconv1 = nn.ConvTranspose2d(self.filters[0], self.filters[1], self.kernels[0], stride=self.stride[0],
                                          padding=self.pad[0])
        self.deconv2 = nn.ConvTranspose2d(self.filters[1], self.filters[2], self.kernels[1], stride=self.stride[1],
                                          padding=self.pad[1])
        self.deconv3 = nn.ConvTranspose2d(self.filters[2], self.filters[3], self.kernels[2], stride=self.stride[2],
                                          padding=self.pad[2])
        self.out = nn.ConvTranspose2d(self.filters[-1], 1, self.kernels[-1], stride=self.stride[-1],
                                       padding=self.pad[-1])

    def forward(self, x):
        x = self.latent(x)
        n =  x.shape[0]
        x = x.view(n, 128, 8, 8)
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.deconv2(x))
        x = nn.functional.relu(self.deconv3(x))
        out = torch.sigmoid(self.out(x))
        return out

class ConvVae(nn.Module):
    def __init__(self, latent_dim, params=params):
        super().__init__()
        self.params = params
        self.latent_dim = latent_dim

        self.device = params['device']
        self.encoder = Encoder(latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim).to(self.device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        rec_x = self.decoder(x)
        return rec_x
        # mu, logvar = self.encoder(x)
        # z = self.reparameterize(mu, logvar)
        # rec_x = self.decoder(z)
        # return rec_x, mu, logvar

class RecLoss(nn.Module):
    def __init__(self):
        super(RecLoss, self).__init__()
        self.l_mse = nn.MSELoss()

    def forward(self, x, rec_x, mu, logvar):
        l_reg = self.l_mse(x, rec_x)
        l_kdl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return l_reg + l_kdl

def get_batch_indices(indices, batch_size, shuffle=True, seed=None):
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    total_size = len(indices)
    batch_size_ = min(batch_size, total_size)
    residual = total_size % batch_size_
    if residual > 0:
        for_splitting = indices[:-residual]
    else:
        for_splitting = indices
    batches = np.split(for_splitting, (total_size - residual) / batch_size_)
    return batches

def train(dir, dataset_name='dataset', params = params):
    file_ = h5py.File(os.path.join(dir, dataset_name + '.hdf5'), "r")
    features = file_['features'][:, :, :]

    # Split to train and validation
    dataset_size = len(file_['features'])
    split_per = 0.8
    train_indices = np.arange(0, 14000, 1)
    valid_indices = np.arange(14000, dataset_size, 1)

    # Initialize the autoencoder
    conv_vae = ConvVae(params['latent_dim'])

    optimizer = optim.Adam(conv_vae.parameters(), lr=params['learning_rate'])
    # rec_loss = RecLoss()
    l_mse = nn.MSELoss()

    for epoch in range(params['epochs']):
        minibatches = get_batch_indices(train_indices, batch_size=params['batch_size'])
        for minibatch in minibatches:
            x = torch.tensor(np.expand_dims(features[minibatch], axis=1), dtype=torch.float, requires_grad=True)\
                .to(params['device'])
            # rec_x, mu, logvar = conv_vae(x)
            # loss = rec_loss(x, rec_x, mu, logvar)
            rec_x = conv_vae(x)
            loss = l_mse(x, rec_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute train and validation losses
        train_loss = 0
        for minibatch in minibatches:
            x = torch.tensor(np.expand_dims(features[minibatch], axis=1), dtype=torch.float, requires_grad=True).to(
                params['device'])
            # rec_x, mu, logvar = conv_vae(x)
            # loss = rec_loss(x, rec_x, mu, logvar)
            rec_x = conv_vae(x)
            loss = l_mse(x, rec_x)
            train_loss += loss.detach().cpu().numpy()
        print('epoch:', epoch, 'train_loss:', train_loss / len(minibatches))

        valid_loss = 0
        minibatches = get_batch_indices(valid_indices, batch_size=params['batch_size'])
        for minibatch in minibatches:
            x = torch.tensor(np.expand_dims(features[minibatch], axis=1), dtype=torch.float, requires_grad=True).to(
                params['device'])
            rec_x = conv_vae(x)
            loss = l_mse(x, rec_x)
            valid_loss += loss.detach().cpu().numpy()
        print('epoch:', epoch, 'valid_loss:', valid_loss / len(minibatches))
        print('---')

        with open(dir + 'model/' + str(epoch) + '.pkl', 'wb') as file:
            pickle.dump(conv_vae.state_dict(), file)


def plot(x, rec_x):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x, cmap='gray', vmin=np.min(x), vmax=np.max(x))
    ax[1].imshow(rec_x, cmap='gray', vmin=np.min(rec_x), vmax=np.max(rec_x))
    plt.show()


def test_vae(dir, dataset_name='dataset'):
    file_path = dir + 'model/50.pkl'
    with open(file_path, 'rb') as file:
        state_dict = pickle.load(file)

    device = 'cuda'

    latent_dim = LATENT_DIM
    conv_vae = ConvVae(latent_dim)
    conv_vae.load_state_dict(state_dict)

    file_ = h5py.File(os.path.join(dir, dataset_name + '.hdf5'), "r")
    features = file_['features'][:, :, :]
    dataset_size = len(file_['features'])

    for i in range(14000, dataset_size):
        x = np.expand_dims(np.expand_dims(features[i], axis=0), axis=0)
        x = torch.tensor(x, dtype=torch.float, requires_grad=True).to(device)
        # rec_x, _, _ = conv_vae(x)
        rec_x = conv_vae(x)
        plot(features[i], rec_x.detach().cpu().numpy()[0, 0, :, :])


def estimate_normalizer(dir, dataset_name='dataset'):
    file_path = dir + 'model/50.pkl'
    with open(file_path, 'rb') as file:
        state_dict = pickle.load(file)

    device = 'cuda'
    latent_dim = 256
    conv_vae = ConvVae(latent_dim)
    conv_vae.load_state_dict(state_dict)

    file_ = h5py.File(os.path.join(dir, dataset_name + '.hdf5'), "r")
    features = file_['features'][:, :, :]
    dataset_size = len(file_['features'])

    latents = np.zeros((dataset_size, latent_dim))
    for i in range(dataset_size):
        x = np.expand_dims(np.expand_dims(features[i], axis=0), axis=0)
        x = torch.tensor(x, dtype=torch.float).to(device)
        # rec_x, _, _ = conv_vae(x)
        # rec_x = conv_vae(x)
        # Feature(features[i]).plot()
        # x = conv_vae.encoder(x)
        # print(x.detach().cpu().numpy())
        # rec_x = conv_vae.decoder(x)
        # Feature(rec_x.detach().cpu().numpy()[0, 0, :, :]).plot()
        latents[i] = conv_vae.encoder(x).detach().cpu().numpy()

    scaler = StandardScaler()
    scaler.fit(latents)
    with open(dir + 'normalizer.pkl', 'wb') as file:
        pickle.dump(scaler, file)