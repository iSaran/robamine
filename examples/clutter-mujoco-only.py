import argparse
from mujoco_py import load_model_from_path, MjSim
import os
import cv2
import glfw
import OpenGL.GL as gl

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__),
                        "../robamine/envs/assets/xml/robots/clutter.xml")

    model = load_model_from_path(path)
    sim = MjSim(model)

    if sim._render_context_window is None:
        print('window context is none')
    sim.render(mode='window')
    if sim._render_context_window is None:
        print('window context is None again')
    for i in range(3000):

        if i == 500 or i ==1500 or i==2000:
            if sim._render_context_offscreen is None:
                print('context offscreen is none')
            rgb, depth = sim.render(1920, 1080, depth=True, camera_name='xtion', mode='offscreen')
            if sim._render_context_offscreen is None:
                print('context offscreen is none again')

            cv2.imwrite("/home/iason/Desktop/fds/obs_" + str(i) +".png", rgb)
        # if i % 1 == 0:
        #     sim.render(mode='window')
            #cv2.imwrite("/home/iason/Desktop/fds/obs_" + str(i) +".png", rgb)
        #else:
        #    sim.render(mode='window', device_id=0)

        sim.step()
