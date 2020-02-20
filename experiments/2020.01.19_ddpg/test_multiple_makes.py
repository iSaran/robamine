import gym
import yaml
import robamine
import numpy as np
from robamine.envs.clutter_cont import ClutterCont
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
import gc
from time import sleep
import glfw

import OpenGL.GL as gl

def run():
    with open('params.yml', 'r') as stream:
        params = yaml.safe_load(stream)


    env = gym.make(params['env']['name'], params=params['env']['params'])
    for i in range(100):
        del env
        env = gym.make(params['env']['name'], params=params['env']['params'])


def run2():
    with open('params.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    path = "/home/iason/ws/robamine/src/robamine/robamine/envs/assets/xml/robots/clutter.xml"
    model = load_model_from_path(path)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    for i in range(20):
        close_viewer(viewer)
        viewer = MjViewer(sim)


def close_viewer(viewer):
    glfw.make_context_current(viewer.window)
    glfw.destroy_window(viewer.window)


def test_multiple_runs():
    with open('params.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = gym.make(params['env']['name'], params=params['env']['params'])

    # Push target
    # -----------
    for i in range(100):
        print(i)
        action = np.array([0, -1, 0, 0.2])
        env.reset()
        for _ in range(4):
            action[1] += 0.5
            env.step(action)

if __name__ == '__main__':
    test_multiple_runs()
