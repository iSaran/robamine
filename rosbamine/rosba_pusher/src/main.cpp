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

int main(int argc, char** argv)
{
  // Initialize the ROS node
  ros::init(argc, argv, "rosba_object_detector");
  ros::NodeHandle n;

  // Create the robot after you have launch the URDF on the parameter server
  auto model = std::make_shared<arl::robot::ROSModel>();

  // Create a simulated robot
  // auto robot = std::make_shared<arl::robot::RobotSim>(model, 1e-3);
  auto robot = std::make_shared<arl::robot::RobotSim>(model, 1e-3);

  // Create a visualizater for see the result in rviz
  auto rviz = std::make_shared<arl::viz::RosStatePublisher>(robot);

  std::thread rviz_thread(&arl::viz::RosStatePublisher::run, rviz);

  Eigen::VectorXd joint(7);
  joint << 1.3078799282743128, -0.36123428594319007, 0.07002260959000406, 1.2006818056150501, -0.0416365746355698, -1.51290026484531, -1.5423125534021;
  robot->setMode(arl::robot::Mode::POSITION_CONTROL);
  robot->setJointTrajectory(joint, 2);

  ros::spin();
  rviz_thread.join();
  return 0;
}
