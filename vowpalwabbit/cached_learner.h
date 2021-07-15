#pragma once

#include "global_data.h"  // to get vw struct
#include "options.h"      // to get options_i

namespace VW
{
struct cached_learner : public setup_base_fn
{
  VW::LEARNER::base_learner* operator()() override { return _cached; }

  operator bool() const { return !(_cached == nullptr); }

  cached_learner(VW::LEARNER::base_learner* learner = nullptr) : _cached(learner) {}

  VW::config::options_i* get_options() override { return nullptr; }

  vw* get_all_pointer() override { return nullptr; }

private:
  VW::LEARNER::base_learner* _cached = nullptr;
};
}  // namespace VW
