/*******************************************************************************
 * Copyright (c) 2016-2017 Automation and Robotics Lab, AUTh
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

#ifndef RLRL_MULTIVARIATE_NORMAL_POLICY_H
#define RLRL_MULTIVARIATE_NORMAL_POLICY_H

#include <rlrl/stochastic_policy.h>

#include <autharl_core/math/mvn.h>
#include <autharl_core/math/differential.h>

namespace rlrl
{
template<class S, class A>
class MultivariateNormalPolicy : public StochasticPolicy<S, A>
{
public:
  MultivariateNormalPolicy(const std::shared_ptr<State<S>>& s,
                           const std::shared_ptr<Action<A>>& a)
    : StochasticPolicy<S, A>(s, a)
    , mvn(Eigen::VectorXd::Zero(this->action->size()),
          Eigen::MatrixXd::Identity(this->action->size(), this->action->size()))
    , theta(3 * this->state->size())
  {
    theta.setZero();
  }

  ~MultivariateNormalPolicy()
  {
  }

  virtual void calcMean()
  {
    mvn.mean = theta.segment(0, this->state->size()).asDiagonal() * Eigen::Map<Eigen::VectorXd>(this->state->data(), this->state->size());
               theta.segment(this->state->size(), this->state->size());
  }

  virtual void calcSigma()
  {
    mvn.sigma = theta.segment(2 * this->state->size(), this->state->size()).asDiagonal();
  }

  double getProbability()
  {
    calcMean();
    calcSigma();
    return mvn.pdf(Eigen::Map<Eigen::VectorXd>(this->action->data(), this->action->size()));
  }

  Action<A> getAction()
  {
    Eigen::VectorXd eig_result = mvn.sample();
    return Action<A>(eig_result.data(), eig_result.data() + eig_result.rows());
  }

  Eigen::VectorXd calcEligibilityVector()
  {
    return arl::math::grad([this](const Eigen::VectorXd& x)
                      {
                        this->theta = x;
                        return std::log(this->pi());
                      },
                      this->theta);
  }

private:
  arl::math::Mvn mvn;
  Eigen::VectorXd theta;
};
}  // namespace rlrl
#endif  // RLRL_MULTIVARIATE_NORMAL_POLICY_H
