#ifndef STATIC_LINK_VW
#  define BOOST_TEST_DYN_LINK
#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

// this test is a copy from unit_test/prediction_test.cc
// it adds a noop reduction on top

#include "vw.h"
#include "reductions_fwd.h"
#include "reduction_stack.h"

// global var eek!
// but useful for this minimal test
bool did_change = false;
bool did_call_learn_predict = false;

template <bool is_learn>
void predict_or_learn(char&, VW::LEARNER::single_learner& base, example& ec)
{
  did_call_learn_predict = true;

  if (is_learn)
    base.learn(ec);
  else
    base.predict(ec);
}

VW::LEARNER::base_learner* test_reduction(VW::setup_base_i& setup_base, VW::config::options_i& options, vw& all)
{
  auto ret = VW::LEARNER::make_no_data_reduction_learner(as_singleline(setup_base(options, all)),
      predict_or_learn<true>, predict_or_learn<false>, all.get_setupfn_name(test_reduction))
                 .set_learn_returns_prediction(true)
                 .build();

  did_change = true;

  return make_base(*ret);
}

// inherits from the default behaviour stack builder
struct custom_builder : VW::default_reduction_stack_setup
{
  custom_builder() { reduction_stack.emplace_back("my_name", test_reduction); }
};

BOOST_AUTO_TEST_CASE(custom_reduction_test)
{
  float prediction_one;
  {
    std::unique_ptr<VW::setup_base_i> learner_builder = VW::make_unique<custom_builder>();
    assert(did_change == false);
    assert(did_call_learn_predict == false);
    auto& vw = *VW::initialize(
        "--quiet --sgd --noconstant --learning_rate 0.1", nullptr, false, nullptr, nullptr, std::move(learner_builder));
    // auto& vw = *VW::initialize("--quiet --sgd --noconstant --learning_rate 0.1");
    auto& pre_learn_predict_example = *VW::read_example(vw, "0.19574759682114784 | 1:1.430");
    auto& learn_example = *VW::read_example(vw, "0.19574759682114784 | 1:1.430");
    auto& predict_example = *VW::read_example(vw, "| 1:1.0");

    vw.predict(pre_learn_predict_example);
    vw.finish_example(pre_learn_predict_example);
    vw.learn(learn_example);
    vw.finish_example(learn_example);
    vw.predict(predict_example);
    prediction_one = predict_example.pred.scalar;
    vw.finish_example(predict_example);
    VW::finish(vw);
    assert(did_change == true);
    assert(did_call_learn_predict == true);
  }

  // reset for second part of test
  did_change = false;
  did_call_learn_predict = false;

  float prediction_two;
  {
    assert(did_change == false);
    assert(did_call_learn_predict == false);
    auto& vw = *VW::initialize("--quiet --sgd --noconstant --learning_rate 0.1");

    auto& learn_example = *VW::read_example(vw, "0.19574759682114784 | 1:1.430");
    auto& predict_example = *VW::read_example(vw, "| 1:1.0");

    vw.learn(learn_example);
    vw.finish_example(learn_example);
    vw.predict(predict_example);
    prediction_two = predict_example.pred.scalar;
    vw.finish_example(predict_example);
    VW::finish(vw);
    // both should be false since it uses the default stack builder
    assert(did_change == false);
    assert(did_call_learn_predict == false);
  }

  BOOST_CHECK_EQUAL(prediction_one, prediction_two);
}
