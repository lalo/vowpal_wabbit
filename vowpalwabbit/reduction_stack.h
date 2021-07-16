#pragma once

#include "reductions_fwd.h"

struct vw;

typedef VW::LEARNER::base_learner* (*reduction_setup_fn)(VW::setup_base_fn&);

namespace VW
{
struct default_reduction_stack_setup : public setup_base_fn
{
  default_reduction_stack_setup(vw& all, VW::config::options_i& options);

  // this function consumes all the reduction_stack until it's able to construct a base_learner
  // same signature as the old setup_base(...) from parse_args.cc
  VW::LEARNER::base_learner* setup_base_learner() override;

  VW::config::options_i* get_options() override { return options_impl; }

  vw* get_all_pointer() override { return all_ptr; }

private:
  std::vector<std::tuple<std::string, reduction_setup_fn>> reduction_stack;
  VW::config::options_i* options_impl = nullptr;
  vw* all_ptr = nullptr;
};
}  // namespace VW
