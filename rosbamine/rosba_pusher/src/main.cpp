/*******************************************************************************
 * Copyright (c) 2019 Automation and Robotics Lab, AUTh
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 ******************************************************************************/

#include <ros/ros.h>
#include <thread>
#include <autharl_core>
#include <lwr_robot/robot_sim.h>
#include <rosba_msgs/Push.h>
#include <rosba_pusher/controller.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>

#include <cmath>

std::shared_ptr<arl::robot::Robot> robot;
std::shared_ptr<roba::Controller> controller;

const double PUSH_DISTANCE = 0.1;
const unsigned int NR_ROTATIONS = 8;
const unsigned int PRIMITIVES = 3;
const double BOUNDING_BOX_Z = 0.05;
const double BOUNDING_BOX_XY_MAX = 0.05;
const double SURFACE_SIZE = 0.2;
const double PUSH_DURATION = 4;

bool callback(rosba_msgs::Push::Request  &req,
              rosba_msgs::Push::Response &res)
{
  // Move arm to home position
  ROS_INFO("Moving arm to home position");
  Eigen::VectorXd joint(7);
  joint << 1.3078799282743128, -0.36123428594319007, 0.07002260959000406, 1.2006818056150501, -0.0416365746355698, -1.51290026484531, -1.5423125534021;
  robot->setMode(arl::robot::Mode::POSITION_CONTROL);
  robot->setJointTrajectory(joint, 8);

  // Call object detection to publish object pos in TF
  ROS_INFO("Call object detection.");
  ros::Duration(5.0).sleep();

  // Read pose of object from TF
  ROS_INFO("Reading target object pose");
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);
  geometry_msgs::TransformStamped transformStamped;
  int k = 0;
  for (unsigned int i = 0; i < 5; i++)
  {
    try
    {
      transformStamped = tfBuffer.lookupTransform("world", "target_object", ros::Time(0));
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN("%s",ex.what());
      ros::Duration(0.5).sleep();
      k++;
      if (k > 4)
      {
        ROS_ERROR("Object frame transformation wasn't found!!!!");
        res.success = false;
        ROS_INFO("Pusher finished.");
        return false;
      }
      continue;
    }
  }
  Eigen::Affine3d target_frame;
  target_frame.translation() << transformStamped.transform.translation.x, transformStamped.transform.translation.y, transformStamped.transform.translation.z;
  auto quat = Eigen::Quaterniond(transformStamped.transform.rotation.w, transformStamped.transform.rotation.x, transformStamped.transform.rotation.y, transformStamped.transform.rotation.z);
  target_frame.linear() = quat.toRotationMatrix();
  std::cout << target_frame.matrix() << std::endl;

  // Create points w.r.t. {O} for trajectory given pushing primitive
  Eigen::Vector3d push_init_pos, push_final_pos, direction;
  push_init_pos.setZero();
  unsigned int primitive_action = static_cast<unsigned int>(std::floor(req.action / NR_ROTATIONS));
  unsigned int rotation = static_cast<unsigned int>(req.action - primitive_action * NR_ROTATIONS);
  double theta = rotation * 2 * M_PI / NR_ROTATIONS;
  direction << std::cos(theta), std::sin(theta), 0.0;
  switch (primitive_action)
  {
    case 0:
      push_init_pos = -(std::sqrt(std::pow(BOUNDING_BOX_XY_MAX, 2) + std::pow(BOUNDING_BOX_XY_MAX, 2)) + 0.001) * direction;
      break;
    case 1:
      push_init_pos(2) = BOUNDING_BOX_Z;
      break;
    case 2:
      push_init_pos = - SURFACE_SIZE * direction;
      break;
    default:
      ROS_ERROR("Error in pushing primitive id. 0, 1 or 2.");
      res.success = false;
      return false;
  }
  Eigen::Vector4d temp;
  temp << push_init_pos, 1.0;
  temp = target_frame * temp;
  push_init_pos(0) = temp(0);
  push_init_pos(1) = temp(1);
  push_init_pos(2) = temp(2);
  push_final_pos(0) = PUSH_DISTANCE * std::cos(theta);
  push_final_pos(1) = PUSH_DISTANCE * std::sin(theta);
  push_final_pos(2) = push_init_pos(2);
  temp << push_final_pos, 1.0;
  temp = target_frame * temp;
  push_final_pos(0) = temp(0);
  push_final_pos(1) = temp(1);
  push_final_pos(2) = temp(2);

  controller->setParams(PUSH_DURATION, push_init_pos);
  controller->run();

  res.success = true;
  ROS_INFO("Pusher finished.");
  return true;
}

int main(int argc, char** argv)
{
  // Initialize the ROS node
  ros::init(argc, argv, "rosba_push");
  ros::NodeHandle n;
  ROS_INFO("Ready to push objects.");

  // Create the robot after you have launch the URDF on the parameter server
  auto model = std::make_shared<arl::robot::ROSModel>();

  // Create a simulated robot
  // auto robot = std::make_shared<arl::robot::RobotSim>(model, 1e-3);
  robot.reset(new arl::lwr::RobotSim(model, 1e-3));
  std::shared_ptr<arl::robot::Sensor> sensor;

  // Create a visualizater for see the result in rviz
  auto rviz = std::make_shared<arl::viz::RosStatePublisher>(robot);

  std::thread rviz_thread(&arl::viz::RosStatePublisher::run, rviz);

  controller.reset(new roba::Controller(robot, sensor));

  ros::ServiceServer service = n.advertiseService("push", callback);
  ROS_INFO("Ready to push objects.");
  ros::spin();

  rviz_thread.join();
  return 0;
}
