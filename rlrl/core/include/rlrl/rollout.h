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

#ifndef RLRL_ROLLOUT_H
#define RLRL_ROLLOUT_H

#include <rlrl/action.h>
#include <rlrl/state.h>

namespace rlrl
{
template<class S, class A>
struct RolloutInOneStep
{
  Action<A> action;
  State<S>  state;
  double    reward;
  double    exp_return;
};

template<class S, class A>
struct Rollout : std::vector<RolloutInOneStep<S, A>>
{
  using std::vector<RolloutInOneStep<S, A>>::vector;

  void calcExpectedReturn()
  {
    for (unsigned int t = 0; t < this->size(); t++)
    {
      double sum = 0;
      for (unsigned int tn = t; tn < this->size(); tn++)
      {
        sum += this->at(tn).reward;
      }
      this->at(t).exp_return = sum / static_cast<double>(this->size());
    }
  }

  RolloutInOneStep<S, A> operator()(unsigned int index) const
  {
    return this->at(index);
  }
};
}  // namespace rlrl
#endif  // RLRL_ROLLOUT_H
