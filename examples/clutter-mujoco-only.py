import argparse
from mujoco_py import load_model_from_path, MjSim
import os
import cv2

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__),
                        "../robamine/envs/assets/xml/robots/clutter.xml")

    model = load_model_from_path(path)
    sim = MjSim(model)
    for i in range (3000):

        if i == 500 or i ==1500 or i==2000:
            rgb, depth = sim.render(1920, 1080, depth=True, camera_name='xtion', mode='offscreen')
            cv2.imwrite("/home/iason/Desktop/fds/obs.png", rgb)

        sim.render(mode='window')
        sim.step()
