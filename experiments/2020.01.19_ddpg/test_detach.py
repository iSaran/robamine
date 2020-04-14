import gym
import yaml
import robamine
import numpy as np

import torch

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.hidden_layer = torch.nn.Linear(2, 3)
        self.out = torch.nn.Linear(3, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden_layer(x))
        out = self.out(x)
        return out

def run():
    torch.manual_seed(0)
    net = NN()
    target_net = NN()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    print('\n----------- Net: Before update ----------')
    print('net: ', net)
    print('hidden_layers weights: ', net.hidden_layer.weight.data)
    print('hidden_layers bias: ', net.hidden_layer.bias.data)
    print('out weights: ', net.hidden_layer.weight.data)
    print('out bias: ', net.hidden_layer.bias.data)

    print('\n----------- Target Net: Before update ----------')
    print('target_net: ', target_net)
    print('hidden_layers weights: ', target_net.hidden_layer.weight.data)
    print('hidden_layers bias: ', target_net.hidden_layer.bias.data)
    print('out weights: ', target_net.hidden_layer.weight.data)
    print('out bias: ', target_net.hidden_layer.bias.data)

    state = torch.rand((5, 2))
    print('\n----------- Batch sample: ----------')
    print(state)
    reward = torch.rand((5, 2))
    # print('batch_reward', batch_reward)
    # print('batch_target', batch_target)
    # print('batch_true', batch_true)

    target_q = reward + target_net(state).detach()

    q = net(state).detach()
    loss = torch.nn.functional.mse_loss(q, target_q)
    print('\n----------- Loss: ----------')
    print(loss)
    optimizer.zero_grad()

    loss.backward()

    print('\n----------- Gradients of batch_reward: ----------')
    print('grad', state.grad)

    print('\n----------- Gradients of batch_target: ----------')
    print('grad', reward.grad)

    print('\n----------- Gradients of target: ----------')
    print('grad', target_q.grad)

    print('\n----------- Gradients of net: ----------')
    print('grad', net.hidden_layer.weight.data.grad)

    optimizer.step()

    print('\n----------- After update ----------')
    print('hidden_layers weights: ', net.hidden_layer.weight.data)
    print('hidden_layers bias: ', net.hidden_layer.bias.data)
    print('out weights: ', net.hidden_layer.weight.data)
    print('out bias: ', net.hidden_layer.bias.data)

    print('\n----------- Target Net: After update ----------')
    print('target_net: ', target_net)
    print('hidden_layers weights: ', target_net.hidden_layer.weight.data)
    print('hidden_layers bias: ', target_net.hidden_layer.bias.data)
    print('out weights: ', target_net.hidden_layer.weight.data)
    print('out bias: ', target_net.hidden_layer.bias.data)

if __name__ == '__main__':
    run()
