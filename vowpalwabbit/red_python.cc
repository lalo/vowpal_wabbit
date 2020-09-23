// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "red_python.h"

#include "reductions.h"
#include "learner.h"
#include "vw.h"

using namespace LEARNER;
using namespace VW::config;

namespace RED_PYTHON
{
//useful for debugging
void learn(ExternalBinding& external_binding, single_learner& base, example& ec)
{ 
  external_binding.SetBaseLearner(&base);
  external_binding.ActualLearn(&ec);
}

//useful for debugging
void predict(ExternalBinding& external_binding, single_learner& base, example& ec) {
  external_binding.SetBaseLearner(&base);
  external_binding.ActualPredict(&ec);
}

void finish_example(vw& all, ExternalBinding& external_binding, example& ec) {
  external_binding.ActualFinishExample(&ec);
  // have to bubble this out to python?
  VW::finish_example(all, ec);
}

}  // namespace RED_PYTHON
using namespace RED_PYTHON;
VW::LEARNER::base_learner* red_python_setup(options_i& options, vw& all)
{
  if (!all.ext_binding)
    return nullptr;

  all.ext_binding->SetRandomNumber(4);

  auto base = as_singleline(setup_base(options, all));

  VW::LEARNER::learner<ExternalBinding, example>& ret =
      learner<ExternalBinding, example>::init_learner(all.ext_binding.get(), base, learn, predict, 1, base->pred_type);

  if (all.ext_binding->ShouldRegisterFinishExample())
    ret.set_finish_example(finish_example);

  // learner should delete ext_binding
  all.ext_binding.release();

  return make_base(ret);
}

using namespace RED_PYTHON;
VW::LEARNER::base_learner* red_python_base_setup(options_i& options, vw& all)
{
  //VW::LEARNER::learner<ExternalBinding, example>& ret = init_learner(all.ext_binding.get(), learn, predict, ((uint64_t)1 << all.weights.stride_shift()));
  //VW::LEARNER::learner<ExternalBinding, example>& ret =
  //    learner<ExternalBinding, example>::init_learner(all.ext_binding.get(), nullptr, learn, predict, 1, prediction_type_t::scalar);

  if (!all.ext_binding)
    return nullptr;

  //en parte de pyvw.cc agregar scoped calloc or throw porque esta cosa uqiere freeptr
  all.ext_binding->SetRandomNumber(4);

  learner<ExternalBinding, example>& l = init_learner(all.ext_binding.get(), learn, predict, 1);
  all.ext_binding.release();

  return make_base(l);

  //return nullptr;
}