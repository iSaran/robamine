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

#ifndef RLRL_MDP_H
#define RLRL_MDP_H

#include <rlrl/action.h>
#include <rlrl/state.h>
#include <rlrl/reward.h>
#include <memory>

namespace rlrl
{
template<class S, class A>
class MDP
{
public:
  MDP(const std::shared_ptr<State<S>>& s,
      const std::shared_ptr<Action<A>>& a,
      const std::shared_ptr<Reward<S, A>>& r,
      unsigned int ep_duration)
    : state(s)
    , action(a)
    , reward(r)
    , episode_duration(ep_duration)
  {
  }

  ~MDP()
  {
  }

  std::shared_ptr<State<S>> state;
  std::shared_ptr<Action<A>> action;
  std::shared_ptr<Reward<S, A>> reward;
  unsigned int episode_duration;
};
}  // namespace rlrl
#endif  // RLRL_MDP_H
