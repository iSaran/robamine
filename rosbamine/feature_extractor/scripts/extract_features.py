#!/usr/bin/env python2

# ros
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2
from std_srvs.srv import Trigger, TriggerRequest
from rosba_msgs.srv import ExtractFeatures
import tf2_ros

import numpy as np

# cv_tools
import cv_tools as cv

import orientation as ori

def callback(req):
    # target objcet dimensions
    b = [0.025, 0.025, 0.025]

    print('===================')
    # get the point cloud
    pcd2 = rospy.wait_for_message("/rosba_pc", PointCloud2)
    print('===================')

    pcd = []
    for point in sensor_msgs.point_cloud2.read_points(pcd2, skip_nans=True):
        pcd.append([point[0], point[1], point[2]])

    point_cloud = np.asarray(pcd)
    # cv.plot_point_cloud(point_cloud)



    # get the target pose transform from tf
    tfBuffer = tf2_ros.Buffer()
    # listener = tf.TransformListener()
    listener = tf2_ros.TransformListener(tfBuffer)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            transform = tfBuffer.lookup_transform('target_object', 'asus_xtion_depth_optical_frame', rospy.Time(0))
            break;
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue


    trans = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
    quat = [transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z]
    rot_mat = ori.quat2rot(quat)
    camera_to_target = np.eye(4, dtype=np.float32)
    camera_to_target[0:3, 0:3] = rot_mat
    camera_to_target[0:3, 3] = trans

    # print(camera_to_target)
    # raw_input('')

    # transform cloud w.r.t. to target
    point_cloud = cv.transform_point_cloud(point_cloud, camera_to_target)
    # cv.plot_point_cloud(point_cloud)

    # translate the point cloud by +b3
    # target_to_table = np.eye(4, dtype=np.float32)
    # target_to_table[3,3] = b[2]
    # point_cloud = cv.transform_point_cloud(point_cloud, target_to_table)

    # Keep the points above the table
    z = point_cloud[:, 2]
    ids = np.where((z > 0.02) & (z < 0.4))
    points_above_table = point_cloud[ids]
    # cv.plot_point_cloud(points_above_table)

    # max height of the scene(max height of obstacles)
    max_height = np.max(z)
    print("max_z:", np.max(z))

    # generate rotated heightmaps
    heightmaps = cv.generate_height_map(points_above_table, rotations=8, plot=False)

    # compute the distances from the table limits (and normalize them)


    while not rospy.is_shutdown():
        try:
            tff = tfBuffer.lookup_transform('world', 'target_object', rospy.Time(0))
            break;
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

    table_limits_x = []
    table_limits_y = []

    table_center = np.array([0.2, 0.65])
    surface_size = np.array([0.25, 0.25])

    print('tff:', tff.transform.translation.x, tff.transform.translation.y)

    obj = [tff.transform.translation.x - table_center[0], tff.transform.translation.y - table_center[1]];

    print('obj:', obj)


    # Add the distance of the object from the edge
    distances = [surface_size[0]  - obj[0], \
                 surface_size[0] + obj[0], \
                 surface_size[1] - obj[1], \
                 surface_size[1] + obj[1]]

    print('distances withoud nrom: ', distances)

    print(2*surface_size[0])
    distances = [(x / (2*surface_size[0])) for x in distances]

    print('distances: ', distances)

    features = []
    rot_angle = 360 / 8.0
    for i in range(0, len(heightmaps)):
        f = cv.extract_features(heightmaps[i], b, max_height, rotation_angle=i*rot_angle, plot=False)
        fff = np.array(f)
        print('i = ', i, np.where(fff > 1))
        f.append(i*rot_angle)
        f.append(b[0]/0.03)
        f.append(b[1]/0.03)
        f.append(distances[0])
        f.append(distances[1])
        f.append(distances[2])
        f.append(distances[3])
        features.append(f)

    final_feature = np.append(features[0], features[1], axis=0)
    for i in range(2, len(features)):
        final_feature = np.append(final_feature, features[i], axis=0)

    rospy.loginfo('Feature extraced. Waiting for new trigger...')
    return final_feature



if __name__ == '__main__':
    rospy.init_node('extract_features_server')
    s = rospy.Service('extract_features', ExtractFeatures, callback)
    rospy.spin()
