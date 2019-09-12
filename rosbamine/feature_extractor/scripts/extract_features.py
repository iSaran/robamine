#!/usr/bin/env python2

# ros
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2
from std_srvs.srv import Trigger, TriggerRequest
import tf

import numpy as np

# cv_tools
import cv_tools as cv

import orientation as ori

def callback(req):
    # target objcet dimensions
    b = [0.05, 0.05, 0.05]

    # get the point cloud
    pcd2 = rospy.wait_for_message("/asus_xtion/depth/points", PointCloud2)
    pcd = []
    for point in sensor_msgs.point_cloud2.read_points(pcd2, skip_nans=True):
        pcd.append([point[0], point[1], point[2]])

    point_cloud = np.asarray(pcd)

    # get the target pose transform from tf
    listener = tf.TransformListener()
    (trans, quat) = listener.lookupTransform('/world', '/target_object', rospy.Time(0))

    rot_mat = ori.quat2rot(quat)
    camera_to_target = np.eye(4, dtype=np.float32)
    camera_to_target[0:3, 0:3] = rot_mat
    camera_to_target[0:3, 3] = trans

    print(camera_to_target)
    raw_input('')

    # transform cloud w.r.t. to target
    point_cloud = cv_tools.transform_point_cloud(point_cloud, camera_to_target)

    # translate the point cloud by +b3
    target_to_table = np.eye(4, dtype=np.float32)
    target_to_table[3,3] = b[2]
    point_cloud = cv_tools.transform_point_cloud(point_cloud, target_to_table)

    # max height of the scene(max height of obstacles)
    max_height = np.max(point_cloud[:, 3])

    # generate rotated heightmaps
    heightmaps = cv.generate_height_map(point_cloud, rotations=8, plot=True)

    # compute the distances from the table limits (and normalize them)
    table_limits_x = []
    table_limits_y = []

    features = []
    rot_angle = 360 / 8.0
    for i in range(0, len(heightmaps)):
        f = cv.extract_features(heightmaps[i], b, max_height, rotation_angle=i*rot_angle, plot=False)
        f.append(i*rot_angle)
        f.append(b[0])
        f.append(b[1])
        f.append(distances[0])
        f.append(distances[1])
        f.append(distances[2])
        f.append(distances[3])
        features.append(f)

    final_feature = np.append(features[0], features[1], axis=0)
    for i in range(2, len(features)):
        final_feature = np.append(final_feature, features[i], axis=0)



if __name__ == '__main__':
    rospy.init_node('extract_features_server')
    s = rospy.Service('extract_features', Trigger, callback)
    rospy.spin()
