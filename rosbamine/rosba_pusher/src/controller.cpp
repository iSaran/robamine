/*******************************************************************************
 * Copyright (c) 2016-2019 Automation and Robotics Lab, AUTh
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

#include <rosba_pusher/controller.h>
#include <autharl_core/math/orientation.h>
#include <autharl_core/math/filters.h>
#include <ros/ros.h>

namespace roba
{

Controller::Controller(const std::shared_ptr<arl::robot::Robot>& arm,
                       const std::shared_ptr<arl::robot::Sensor>& the_sensor)
  : arl::robot::Controller(arm, "roba controller")
  , sensor(the_sensor)
  , push_done(false)
  , stop_if_force(false)
  , traj_interrupted(false)
  , duration(2)
{
  this->arm_pos_init.setZero();
}

void Controller::init()
{
  std::cout << "INITTTA" << std::endl;
  push_done = false;
  traj_interrupted = false;
  this->t = 0.0;
  this->arm_pos_init = robot->getTaskPosition();
  this->arm_quat_d = Eigen::Quaterniond(robot->getTaskOrientation());
  robot->setMode(arl::robot::Mode::TORQUE_CONTROL);
  std::cout << "INITTTA: end" << std::endl;
}

void Controller::setParams(double duration, const Eigen::Vector3d& final_pos, bool stop_if_force)
{
  Eigen::Vector3d init_pos = robot->getTaskPosition();
  this->duration = duration;
  this->traj.setParams(0.0,  init_pos,  Eigen::Vector3d::Zero(),  Eigen::Vector3d::Zero(),
                       duration, final_pos, Eigen::Vector3d::Zero(),  Eigen::Vector3d::Zero());
  this->stop_if_force = stop_if_force;

}

void Controller::update()
{
  std::cout << "UPDATE" << std::endl;
  if (this->t > duration)
  {
    push_done = true;
  }

  // if (sensor->getData().norm() > 10 && this->stop_if_force)
  // {
  //   this->traj_interrupted = true;
  // }

  std::cout << "UPDATE read state" << std::endl;
  // Read state from the robot
  Eigen::Vector3d arm_pos = robot->getTaskPosition();
  Eigen::Matrix3d arm_rot = robot->getTaskOrientation();
  Eigen::Quaterniond arm_quat = Eigen::Quaterniond(arm_rot);
  Eigen::MatrixXd jac = robot->getJacobian();

  // Eigen::Vector3d arm_pos_d = this->traj.pos(this->t);
  Eigen::Vector3d arm_pos_d = this->arm_pos_init;

  Eigen::Vector6d pos_error;
  pos_error.segment(0, 3) = arm_pos - arm_pos_d;
  pos_error.segment(3, 3) = arm_quat.log_error(arm_quat_d);

  // Calculate arm's velocity
  std::cout << "UPDATE calculate velocity " << std::endl;
  Eigen::VectorXd joint_vel = robot->getJointVelocity();
  Eigen::VectorXd vel = jac * joint_vel;

  // # A simple force with variable stiffness
  std::cout << "UPDATE calculate impedance" << std::endl;
  double max_stiff_trans = 500;
  double stiff_rot = 50.0;
  Eigen::Vector6d stiffness;
  stiffness << max_stiff_trans, max_stiff_trans, max_stiff_trans, stiff_rot, stiff_rot, stiff_rot;
  Eigen::Vector6d force = stiffness.asDiagonal() * pos_error;
  Eigen::Vector6d vel_d = Eigen::Vector6d::Zero();
  Eigen::Vector6d vel_error = vel - vel_d;
  double damp_trans = 20.0;
  double damp_rot = 2.0;
  Eigen::Vector6d damping;
  damping << damp_trans, damp_trans, damp_trans, damp_rot, damp_rot, damp_rot;
  force += damping.asDiagonal() * vel_error;

  // # Command the robot
  Eigen::VectorXd arm_commanded_torques = - jac.transpose() * force;
  std::cout << "UPDATE Sending joint torques: " << arm_commanded_torques << std::endl;
  robot->setJointTorque(arm_commanded_torques);
  std::cout << "UPDATE end" << std::endl;
}

bool Controller::success()
{
  return !this->traj_interrupted;
}

bool Controller::stop()
{
  if (this->push_done)
  {
    return true;
  }

  if (this->traj_interrupted)
  {
    return true;
  }

  return !ros::ok();
}
}  // namespace fog
