// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "debug_log.h"
#include "reductions.h"
#include "learner.h"
#include <cfloat>

#include "io/logger.h"

using namespace VW::config;
using namespace VW::LEARNER;

namespace logger = VW::io::logger;

namespace VW
{
namespace test_red
{
struct tr_data
{
  //problem multiplier
  size_t pm = 2;
  std::vector<std::vector<namespace_index>>* backup = nullptr;
  std::vector<std::vector<namespace_index>> interactions_1;
  std::vector<std::vector<namespace_index>> interactions_2;
};

// not implemented yet
template <bool is_learn, typename T>
void predict_or_learn(tr_data& data, T& base, example& ec)
{
  for (uint32_t i = 0; i < data.pm; i++)
  {
    if (is_learn)
    {
      base.learn(ec, i);
    }
    else
    {
      base.predict(ec, i);
    }
  }
}

void configure_interactions(tr_data& data, example* ec, size_t config_number)
{
  if (ec == nullptr) return;
  if (ec->interactions == nullptr) return;

  if (config_number == 1)
  {
    ec->interactions = &(data.interactions_1);
    // std::cerr << config_number << "int:" << ec->interactions <<"s"<< ec->interactions->size() << std::endl;
  }
  else if (config_number == 0)
  {
    // std::cerr << config_number << "int:" << ec->interactions <<"s"<< ec->interactions->size() << std::endl;
  }
}

void restore_interactions(tr_data& data, example* ec, size_t config_number)
{
  if (config_number == 1)
  {
    ec->interactions = data.backup;
  }
}

template <bool is_learn, typename T>
void predict_or_learn_m(tr_data& data, T& base, multi_ex& ec)
{
  for (uint32_t i = 0; i < data.pm; i++)
  {
    if (ec[0]->interactions != nullptr)
    {
      data.backup = ec[0]->interactions;
      for (example* ex : ec) configure_interactions(data, ex, i);
    }

    if (is_learn)
    {
      base.learn(ec, i);
    }
    else
    {
      base.predict(ec, i);
    }

    if (ec[0]->interactions != nullptr)
    {
      for (example* ex : ec) restore_interactions(data, ex, i);

      data.backup = nullptr;
    }
  }
}

void persist(tr_data& , metric_sink& )
{
  // metrics.int_metrics_list.emplace_back("total_predict_calls", data.predict_count);
  // metrics.int_metrics_list.emplace_back("total_learn_calls", data.learn_count);
}

VW::LEARNER::base_learner* test_red_setup(options_i& options, vw& all)
{
  bool test_red;
  auto data = scoped_calloc_or_throw<tr_data>();

  option_group_definition new_options("Debug: test reduction");
  new_options.add(make_option("test_red", test_red)
                      .necessary()
                      .help("blah blah"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  auto* base_learner = setup_base(options, all);

  if (base_learner->is_multiline)
  {
    learner<tr_data, multi_ex>* l = &init_learner(data, as_multiline(base_learner),
        predict_or_learn_m<true, multi_learner>, predict_or_learn_m<false, multi_learner>, data->pm,
        base_learner->pred_type, all.get_setupfn_name(test_red_setup), base_learner->learn_returns_prediction);
    l->set_persist_metrics(persist);
    return make_base(*l);
  }
  else
  {
    learner<tr_data, example>* l = &init_learner(data, as_singleline(base_learner),
        predict_or_learn<true, single_learner>, predict_or_learn<false, single_learner>, data->pm,
        base_learner->pred_type, all.get_setupfn_name(test_red_setup), base_learner->learn_returns_prediction);
    l->set_persist_metrics(persist);
    return make_base(*l);
  }
}

}  // namespace test_red
}  // namespace VW
