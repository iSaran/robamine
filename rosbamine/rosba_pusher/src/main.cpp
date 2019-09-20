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
#include <lwr_robot/robot.h>
#include <rosba_msgs/Push.h>
#include <std_srvs/Trigger.h>
#include <rosba_pusher/controller.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>

#include <cmath>

// Moveit staff
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

std::shared_ptr<arl::robot::Robot> robot;
std::shared_ptr<roba::Controller> controller;

// Parameters
const double SURFACE_SIZE = 0.15;
const double PUSH_DURATION = 3.5;
const double PLANNING_TIME = 5.0;
const bool MOVEIT = true;
const bool REAL_ROBOT = true;

// Global variables
bool already_home = false;
std::string group_name = "lwr_ati_xtion_handle";


geometry_msgs::Pose toROS(const Eigen::Affine3d& input)
{
  geometry_msgs::Pose result;
  result.position.x = input.translation()(0);
  result.position.y = input.translation()(1);
  result.position.z = input.translation()(2);
  auto q = Eigen::Quaterniond(input.linear().matrix());
  result.orientation.x = q.x();
  result.orientation.y = q.y();
  result.orientation.z = q.z();
  result.orientation.w = q.w();
  return result;
}

bool callback(rosba_msgs::Push::Request  &req,
              rosba_msgs::Push::Response &res)
{
  already_home = false;
  Eigen::Vector3d arm_init_pos = robot->getTaskPosition();

  // Read pose of object from TF
  ROS_INFO("Reading target object pose");
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);
  geometry_msgs::TransformStamped transformStamped;
  int k = 0;
  for (unsigned int i = 0; i < 10; i++)
  {
    try
    {
      transformStamped = tfBuffer.lookupTransform("world", "target_object", ros::Time(0), ros::Duration(10));
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN("%s",ex.what());
      ros::Duration(0.5).sleep();
      k++;
      if (k > 10)
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

  unsigned int nr_primitives;
  if (req.extra_primitive)
  {
    nr_primitives = 3;
  }
  else
  {
    nr_primitives = 2;
  }

  std::cout << "nr_primitives: " << nr_primitives << std::endl;

  unsigned int nr_rotations = req.nr_actions / nr_primitives;
  std::cout << "nr_rotations: " << nr_rotations << std::endl;


  unsigned int primitive_action = static_cast<unsigned int>(std::floor(req.action / nr_rotations));
  unsigned int rotation = static_cast<unsigned int>(req.action - primitive_action * nr_rotations);
  double theta = rotation * 2 * M_PI / nr_rotations;
  direction << std::cos(theta), std::sin(theta), 0.0;
  switch (primitive_action)
  {
    case 0:
      push_init_pos = -(std::sqrt(std::pow(req.bounding_box.x, 2) + std::pow(req.bounding_box.x, 2)) + 0.001) * direction;
      ROS_INFO_STREAM("PUSH TARGET with theta = " << theta);
      break;
    case 1:
      push_init_pos(2) = req.bounding_box.z;
      ROS_INFO_STREAM("PUSH OBSTACLE with theta = " << theta);
      break;
    case 2:
      push_init_pos = - SURFACE_SIZE * direction;
      ROS_INFO_STREAM("EXTRA with theta = " << theta);
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
  push_final_pos(0) = req.distance * std::cos(theta);
  push_final_pos(1) = req.distance * std::sin(theta);
  push_final_pos(2) = push_init_pos(2);
  temp << push_final_pos, 1.0;
  temp = target_frame * temp;
  push_final_pos(0) = temp(0);
  push_final_pos(1) = temp(1);
  push_final_pos(2) = temp(2);


  if (MOVEIT)
  {
    // Create group_name
    moveit::planning_interface::MoveGroupInterface group(group_name);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    moveit_msgs::RobotState start_state;
    group.setPlanningTime(PLANNING_TIME);
    group.setNumPlanningAttempts(30);
    group.setGoalTolerance(0.005);

    // Create joint trajectory controlle for executing plans
    robot->setMode(arl::robot::Mode::POSITION_CONTROL);
    arl::controller::JointTrajectory trajectory_controller(robot);
    trajectory_msgs::JointTrajectory final_trajectory;

    // Go above the scene
    Eigen::Affine3d pose;
    pose.linear() << 1,  0,  0, 0, -1,  0, 0,  0, -1;
    pose.translation() = push_init_pos;
    pose.translation()(2) += 0.1;
    group.setPoseTarget(toROS(pose));

    ROS_INFO("Planning above push init pose");
    bool success = (group.plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    arl::primitive::JointTrajectory trajectory0(plan.trajectory_.joint_trajectory);
    trajectory0.scale(2 * PUSH_DURATION / trajectory0.duration());
    // ROS_INFO_STREAM("Plan started from: ");
    // for (unsigned int i = 0; i < 7; i++)
    // {
    //   std::cout << plan.start_state_.joint_state.position.at(i) << ",";
    // }
    // std::cout << std::endl;
    //
    // ROS_INFO_STREAM("Plan ended to: " );
    // for (unsigned int i = 0; i < 7; i++)
    // {
    //   std::cout << plan.trajectory_.joint_trajectory.points[plan.trajectory_.joint_trajectory.points.size() - 1].positions.at(i) << ",";
    // }
    // std::cout << std::endl;
    // std::cout << "iwill reach: " << final_trajectory.points[final_trajectory.points.size() - 1].positions << std::endl;


    ROS_INFO("Planning push init pose");
    start_state.joint_state.name =  plan.trajectory_.joint_trajectory.joint_names;
    start_state.joint_state.position = plan.trajectory_.joint_trajectory.points[plan.trajectory_.joint_trajectory.points.size() - 1].positions;
    group.setStartState(start_state);
    pose.translation()(2) -= 0.1;
    if (success)
    {
      group.setPoseTarget(toROS(pose));
      success = (group.plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    }
    else
    {
        ROS_ERROR_STREAM("Moveit failed to produce plan...");
        return false;
    }
    arl::primitive::JointTrajectory trajectory1(plan.trajectory_.joint_trajectory);
    trajectory1.scale(PUSH_DURATION / trajectory1.duration());
    // ROS_INFO_STREAM("Plan started from: ");
    // for (unsigned int i = 0; i < 7; i++)
    // {
    //   std::cout << plan.start_state_.joint_state.position.at(i) << ",";
    // }
    // std::cout << std::endl;
    //
    // ROS_INFO_STREAM("Plan ended to: " );
    // for (unsigned int i = 0; i < 7; i++)
    // {
    //   std::cout << plan.trajectory_.joint_trajectory.points[plan.trajectory_.joint_trajectory.points.size() - 1].positions.at(i) << ",";
    // }
    // std::cout << std::endl;

    ROS_INFO("Planning push final pose");
    start_state.joint_state.name =  plan.trajectory_.joint_trajectory.joint_names;
    start_state.joint_state.position = plan.trajectory_.joint_trajectory.points[plan.trajectory_.joint_trajectory.points.size() - 1].positions;
    group.setStartState(start_state);
    pose.translation() = push_final_pos;
    pose.translation()(2) -= 0.02;
    if (success)
    {
      group.setPoseTarget(toROS(pose));
      success = (group.plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    }
    else
    {
        ROS_ERROR_STREAM("Moveit failed to produce plan...");
        return false;
    }
    arl::primitive::JointTrajectory trajectory2(plan.trajectory_.joint_trajectory);
    trajectory2.scale(PUSH_DURATION / trajectory2.duration());

    ROS_INFO("Planning above push final pose");
    start_state.joint_state.name =  plan.trajectory_.joint_trajectory.joint_names;
    start_state.joint_state.position = plan.trajectory_.joint_trajectory.points[plan.trajectory_.joint_trajectory.points.size() - 1].positions;
    group.setStartState(start_state);
    pose.translation()(2) += 0.2;
    if (success)
    {
      group.setPoseTarget(toROS(pose));
      success = (group.plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
      for (unsigned int i = 0; i < plan.trajectory_.joint_trajectory.points.size(); i++)
      {
        final_trajectory.points.push_back(plan.trajectory_.joint_trajectory.points[i]);
      }
    }
    else
    {
        ROS_ERROR_STREAM("Moveit failed to produce plan...");
        return false;
    }
    arl::primitive::JointTrajectory trajectory3(plan.trajectory_.joint_trajectory);
    trajectory3.scale(PUSH_DURATION / trajectory3.duration());

    // Compute path with waypoints
    // std::vector<geometry_msgs::Pose> waypoints;
    // waypoints.push_back(toROS(pose));
    // pose.translation()(2) -= 0.1;
    // waypoints.push_back(toROS(pose));
    // pose.translation() = push_final_pos;
    // std::cout << "push_final_pose:" << push_final_pos << std::endl;
    // waypoints.push_back(toROS(pose));
    // pose.translation()(2) += 0.1;
    // waypoints.push_back(toROS(pose));
    // moveit_msgs::RobotTrajectory traj;
    // double perc = group.computeCartesianPath(waypoints, 0.005, 0, traj);
    // if (perc < 0.1)
    // {
    //   ROS_ERROR_STREAM("Moveit failed to produce plan for this push. Percentage of trajectory completion is: " << perc);
    //   return false;
    // }

    // Execute plan
    // arl::primitive::JointTrajectory trajectory(my_plan.trajectory_.joint_trajectory);
    trajectory_controller.reference(trajectory0);
    trajectory_controller.run();
    trajectory_controller.reference(trajectory1);
    trajectory_controller.run();
    trajectory_controller.reference(trajectory2);
    trajectory_controller.run();
    trajectory_controller.reference(trajectory3);
    trajectory_controller.run();



    // // group.setMaxVelocityScalingFactor(1);



    // Define constraints
/////////////moveit_msgs::PositionConstraint con;
    /////////con.link_name = "handle_tool";
    /////////con.target_point_offset.x = 0.0;
    /////////con.target_point_offset.y = 0.0;
    /////////con.target_point_offset.z = 0.0;
    /////////shape_msgs::SolidPrimitive prim;
    /////////prim.type = shape_msgs::SolidPrimitive::CYLINDER;
    /////////prim.dimensions.push_back(req.distance);
    /////////prim.dimensions.push_back(0.005);
    /////////con.constraint_region.primitives.push_back(prim);
    /////////con.constraint_region.pose.push_back(prim)



    // Send the planned trajectory to the robot
  }
  else
  {
    Eigen::VectorXd joint(7);
    joint << 1.3078799282743128, -0.36123428594319007, 0.07002260959000406, 1.2006818056150501, -0.0416365746355698, -1.51290026484531, -1.5423125534021;
    // joint << 1.0425608158111572, -0.17295242846012115, 0.41580450534820557, 1.1572487354278564, -0.04072001576423645, -1.6935195922851562, -1.5933283567428589;
    robot->setMode(arl::robot::Mode::POSITION_CONTROL);
    ros::Duration(1.0).sleep();
    robot->setJointTrajectory(joint, 8);
    ros::Duration(1.0).sleep();

    push_init_pos(2) += 0.2;
    ros::Duration(0.5).sleep();
    controller->setParams(PUSH_DURATION, push_init_pos);
    if (!controller->run())
    {
      res.success = false;
      ROS_ERROR("Pusher failed.");
      return false;
    }
    ros::Duration(0.5).sleep();
    push_init_pos(2) -= 0.2;
    controller->setParams(PUSH_DURATION, push_init_pos);
    if (!controller->run())
    {
      res.success = false;
      ROS_ERROR("Pusher failed.");
      return false;
    }
    ros::Duration(0.5).sleep();
    controller->setParams(PUSH_DURATION, push_final_pos);
    if (!controller->run())
    {
      res.success = false;
      ROS_ERROR("Pusher failed.");
      return false;
    }
    ros::Duration(0.5).sleep();
    push_final_pos(2) += 0.2;
    controller->setParams(PUSH_DURATION, push_final_pos);
    if (!controller->run())
    {
      res.success = false;
      ROS_ERROR("Pusher failed.");
      return false;
    }
    ros::Duration(0.5).sleep();
    controller->setParams(PUSH_DURATION, arm_init_pos);
    if (!controller->run())
    {
      res.success = false;
      ROS_ERROR("Pusher failed.");
      return false;
    }
  }

  res.success = true;
  ROS_INFO("Pusher finished.");
  return true;
}

bool goHome(std_srvs::Trigger::Request  &req,
            std_srvs::Trigger::Response &res)
{
  if (already_home)
  {
    ROS_INFO("Arm already home");
    res.success = true;
    return true;
  }

  // Move arm to home position
  ROS_INFO("Moving arm to home position");
  Eigen::VectorXd joint(7);
  // joint << 1.3078799282743128, -0.36123428594319007, 0.07002260959000406, 1.2006818056150501, -0.0416365746355698, -1.51290026484531, -1.5423125534021;
  joint << 1.0425608158111572, -0.17295242846012115, 0.41580450534820557, 1.1572487354278564, -0.04072001576423645, -1.6935195922851562, -1.5933283567428589;


  robot->setMode(arl::robot::Mode::POSITION_CONTROL);
  ros::Duration(1.0).sleep();
  robot->setJointTrajectory(joint, 8);

  already_home = true;

  res.success = true;
  ROS_INFO("Pusher finished.");
  return true;
}


int main(int argc, char** argv)
{
  // Initialize the ROS node
  ros::init(argc, argv, "rosba_pusher");
  ros::NodeHandle n;

  // Start an async spinner for MoveIt!
  ros::AsyncSpinner spinner(2);
  spinner.start();

  // Create the robot after you have launch the URDF on the parameter server
  auto model = std::make_shared<arl::robot::ROSModel>();

  // Create a simulated robot
  // auto robot = std::make_shared<arl::robot::RobotSim>(model, 1e-3);

  if (REAL_ROBOT)
  {
    robot.reset(new arl::lwr::Robot(model));
  }
  else
  {
    robot.reset(new arl::lwr::RobotSim(model, 1e-3));
  }

  std::shared_ptr<arl::robot::Sensor> sensor;

  // Create a visualizater for see the result in rviz
  auto rviz = std::make_shared<arl::viz::RosStatePublisher>(robot);

  std::thread rviz_thread(&arl::viz::RosStatePublisher::run, rviz);

  controller.reset(new roba::Controller(robot, sensor));

  ROS_INFO("Moving arm to home position");
  Eigen::VectorXd joint(7);
  // joint << 1.3078799282743128, -0.36123428594319007, 0.07002260959000406, 1.2006818056150501, -0.0416365746355698, -1.51290026484531, -1.5423125534021;
  joint << 1.0425608158111572, -0.17295242846012115, 0.41580450534820557, 1.1572487354278564, -0.04072001576423645, -1.6935195922851562, -1.5933283567428589;
  robot->setMode(arl::robot::Mode::POSITION_CONTROL);
  robot->setJointTrajectory(joint, 8);

  ros::ServiceServer service = n.advertiseService("push", callback);
  ros::ServiceServer go_home_srv = n.advertiseService("go_home", goHome);
  ROS_INFO("Ready to push objects.");

  rviz_thread.join();
  return 0;
}
