// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include <vector>
#include "reductions_fwd.h"

VW::LEARNER::base_learner* red_python_setup(VW::config::options_i& options, vw& all);
VW::LEARNER::base_learner* red_python_base_setup(VW::config::options_i& options, vw& all);

namespace RED_PYTHON
{
    //this should live elsewhere?
    class ExternalBinding
    {
        public:
            //not needed added for debugging
            virtual void SetRandomNumber(int n) = 0;
            virtual void SetBaseLearner(void* learner) = 0;
            virtual void ActualLearn(example *) = 0;
            virtual void ActualPredict(example *) = 0;
            virtual void ActualSaveLoad() = 0;
            virtual bool ShouldRegisterFinishExample() = 0;
            virtual bool ShouldRegisterSaveLoad() = 0;
            virtual void ActualFinishExample(example *) = 0;
            virtual ~ExternalBinding() {};
    };
}  // namespace RED_PYTHON
