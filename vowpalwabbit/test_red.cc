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
  // all is not needed but good to have for testing purposes
  vw* all;
  // problem multiplier
  size_t pm = 1;
  // backup of the original interactions that come from example parser probably
  std::vector<std::vector<namespace_index>>* backup = nullptr;
  std::vector<std::vector<namespace_index>> interactions_1;
  std::vector<std::vector<namespace_index>> empty_interactions;
};

// see predict_or_learn_m,
// this one not impl yet
template <bool is_learn, typename T>
void predict_or_learn(tr_data& data, T& base, example& ec)
{
  for (uint32_t i = 0; i < data.pm; i++)
  {
    if (is_learn) { base.learn(ec, i); }
    else
    {
      // base.predict(ec, i);
      THROW("not implemented yet");
    }
  }
}

void configure_interactions(tr_data& data, example* ec, size_t config_number)
{
  if (ec == nullptr) return;
  if (ec->interactions_ == nullptr) return;

  if (config_number == 0)
  {
    ec->interactions_ = &(data.interactions_1);
    // std::cerr << config_number << "int:" << ec->interactions <<"s"<< ec->interactions->size() << std::endl;
  }
  else if (config_number == 1)
  {
    // std::cerr << config_number << "int:" << ec->interactions <<"s"<< ec->interactions->size() << std::endl;
  }
}

void restore_interactions(tr_data& data, example* ec, size_t config_number)
{
  if (config_number == 0) { ec->interactions_ = data.backup; }
}

// for debugging purposes
void print_interactions(example* ec)
{
  if (ec == nullptr) return;
  if (ec->interactions_ == nullptr) return;

  if (ec->interactions_->size()) std::cerr << "p:" << ec->interactions_;

  for (std::vector<namespace_index> v : *(ec->interactions_))
  {
    for (namespace_index c : v) { std::cerr << "interaction:" << c << std::endl; }
  }
}

// useful to understand what namespaces are used in the examples we are given
// this can evolve to feed in data to generate possible interactions
void print_all_namespaces_in_examples(multi_ex& exs)
{
  for (example* ex : exs)
  {
    for (auto i : ex->indices) { std::cerr << i << ", "; }
    std::cerr << std::endl;
  }
}

// add an interaction to an existing instance
void add_interaction(
    std::vector<std::vector<namespace_index>>& interactions, namespace_index first, namespace_index second)
{
  std::vector<namespace_index> vect;
  vect.push_back(first);
  vect.push_back(second);
  interactions.push_back(vect);
}

template <bool is_learn, typename T>
void predict_or_learn_m(tr_data& data, T& base, multi_ex& ec)
{
  // extra assert just bc
  assert(data.all->_interactions.empty() == true);

  // we force parser to set always as nullptr, see change in parser.cc
  assert(ec[0]->interactions_ == nullptr);
  // that way we can modify all.interactions without parser caring
  if (ec[0]->interactions_ == nullptr)
  { 
    data.backup = &data.empty_interactions;
    // ec[0]->interactions_ = &data.empty_interactions;
    for (example* ex : ec) { restore_interactions(data, ex, 0); }
    data.backup = nullptr;
  }
  else
  {
    data.backup = ec[0]->interactions_;
  }

  // test this works if interactions turns out to be nullptr
  for (uint32_t i = 0; i < data.pm; i++)
  {
    assert(ec[0]->interactions_ != nullptr);
    assert(data.backup == nullptr);

    for (example* ex : ec) { configure_interactions(data, ex, 1); }

    auto restore_guard = VW::scope_exit([&data, &ec, &i] {
      for (example* ex : ec) { restore_interactions(data, ex, 0); }
      data.backup = nullptr;
    });

    if (is_learn) { base.learn(ec, i); }
    else
    {
      base.predict(ec, i);
    }
  }
}

void persist(tr_data&, metric_sink&)
{
  // metrics.int_metrics_list.emplace_back("total_predict_calls", data.predict_count);
  // metrics.int_metrics_list.emplace_back("total_learn_calls", data.learn_count);
}

VW::LEARNER::base_learner* test_red_setup(options_i& options, vw& all)
{
  bool test_red;
  auto data = scoped_calloc_or_throw<tr_data>();

  option_group_definition new_options("Debug: test reduction");
  new_options.add(make_option("test_red", test_red).necessary().help("blah blah"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  // all is not needed but good to have for testing purposes
  data->all = &all;

  // override and clear all the global interactions
  // see parser.cc line 740
  all._interactions.clear();

  // make sure we setup the rest of the stack with cleared interactions
  // to make sure there are not subtle bugs
  auto* base_learner = setup_base(options, all);

  // hard code test 312 from RunTests
  // if we comment this following line it works ?? why ??
  // add_interaction(all._interactions, 'G', 'T');
  assert(all._interactions.empty() == true);
  add_interaction(data->empty_interactions, 'G', 'T');

  // fail if incompatible reductions got setup
  // inefficient, address later
  // references global all interactions
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"ccb_explore_adf")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"audit_regressor")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"baseline")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"cb_explore_adf_rnd")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"cb_to_cb_adf")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"cbify")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"replay_c")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"replay_b")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"replay_m")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  // if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"gd")!=all.enabled_reductions.end()) THROW("plz no gd");
  // if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"generate_interactions")!=all.enabled_reductions.end()) THROW("plz no gd");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"memory_tree")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"new_mf")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"nn")!=all.enabled_reductions.end()) THROW("plz no bad stack");
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(),"stage_poly")!=all.enabled_reductions.end()) THROW("plz no bad stack");

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
