/*******************************************************************************
 * Copyright (c) 2017-2019 Iason Sarantopoulos, Automation and Robotics Lab, AUTh
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

#ifndef FOG_REACH_FGS_GAMMA_H
#define FOG_REACH_FGS_GAMMA_H

#include <autharl_core>
#include <thread>

namespace roba
{
class Controller : public arl::robot::Controller
{
public:
  explicit Controller(const std::shared_ptr<arl::robot::Robot>& arm,
                      const std::shared_ptr<arl::robot::Sensor>& sensor);
  void init();
  void update();
  bool stop();
  void reset();
  void setParams(double duration, const Eigen::Vector3d& final_pos, bool stop_if_force = false);
  bool success();
private:
  double duration;
  std::shared_ptr<arl::robot::Sensor> sensor;
  Eigen::Quaterniond arm_quat_d;
  arl::primitive::Trajectory<Eigen::Vector3d> traj;
  bool push_done, traj_interrupted;
  bool stop_if_force;

};
}  // namespace roba

#endif  // FOG_ROBOT_CONTROLLER_REACH_FGS_GAMMA_H
