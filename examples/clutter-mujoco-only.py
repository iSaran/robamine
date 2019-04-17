import argparse
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
import os
import cv2
import glfw
import OpenGL.GL as gl

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__),
                        "../robamine/envs/assets/xml/robots/clutter.xml")

    model = load_model_from_path(path)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    off_context = MjRenderContextOffscreen(sim, 0)
    #off_context = MjRenderContextOffscreen(sim, device_id=0)

    for i in range(3000):

        if i == 500 or i ==1500 or i==2000:
            off_context.render(1920, 1080, 0)
            rgb = off_context.read_pixels(1920, 1080)[0]
            # rgb, depth = sim.render(1920, 1080, depth=True, camera_name='xtion', mode='offscreen')
            cv2.imwrite("/home/mkiatos/Desktop/fds/obs_" + str(i) +".png", rgb)
        viewer.render()
        sim.step()
