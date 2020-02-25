import numpy as np
import imageio
import matplotlib.pyplot as plt
from robamine.envs.clutter_utils import TargetObjectConvexHull
from math import pi
from time import sleep

def run():
    mask = imageio.imread('custom_mask_2.png')
    mask = mask[:, :, 0]
    plt.imshow(mask)
    plt.show()
    target_object = TargetObjectConvexHull(mask)
    target_object.plot()
    target_object = target_object
    print('height', target_object.get_bounding_box())

    angle = -(pi / 180) *150
    print(target_object.get_limit_intersection(angle))


    target_object.plot()
    plt.show()

def run2():
    # mask = imageio.imread('custom_mask.png')
    masks = []
    # masks.append(imageio.imread('custom_mask_1.png'))
    # masks.append(imageio.imread('custom_mask_2.png'))
    # masks.append(imageio.imread('custom_mask_3.png'))
    # masks.append(imageio.imread('custom_mask_4.png'))
    # masks.append(imageio.imread('custom_mask_5.png'))
    masks.append(imageio.imread('custom_mask_8.png'))
    for mask in masks:

        mask = mask[:, :, 0]
        target_object = TargetObjectConvexHull(mask)
        target_object.plot(blocking=False)
        target_object.enforce_number_of_points(10)

        target_object.plot(blocking=False)
    plt.show()

if __name__ == '__main__':
    run2()
