#pragma once
// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "reductions_fwd.h"

namespace VW
{
namespace test_red
{
VW::LEARNER::base_learner* test_red_setup(VW::config::options_i& options, vw& all);
}  // namespace test_red
}  // namespace VW