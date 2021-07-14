// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "debug_log.h"
#include "reductions.h"
#include "learner.h"
#include <cfloat>

#include "distributionally_robust.h"
#include "io/logger.h"
#include "vw.h"

using namespace VW::config;
using namespace VW::LEARNER;

namespace logger = VW::io::logger;

namespace VW
{
namespace test_red
{
struct tr_data
{
  tr_data()
      : chisq_1(0.05, 0.999, 0, std::numeric_limits<double>::infinity())
      , chisq_2(0.05, 0.999, 0, std::numeric_limits<double>::infinity())
  {
  }
  size_t which_to_return = 0;
  size_t county = 0;
  // all is not needed but good to have for testing purposes
  vw* all;
  // problem multiplier
  size_t pm = 2;
  // to simulate printing in cb_explore_adf
  multi_learner* adf_learner;
  ACTION_SCORE::action_scores a_s;  // a sequence of classes with scores.  Also used for probabilities.
  // backup of the original interactions that come from example parser probably
  std::vector<std::vector<namespace_index>>* backup = nullptr;
  std::vector<std::vector<namespace_index>> interactions_1;
  std::vector<std::vector<namespace_index>> empty_interactions;

  VW::distributionally_robust::ChiSquared chisq_1;
  VW::distributionally_robust::ChiSquared chisq_2;
  float ipsone = 0.0;
  float ipstwo = 0.0;

  float w1 = 0.0;
  float w2 = 0.0;
  float r1 = 0.0;
  float r2 = 0.0;
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
  // if (ec->interactions_ == nullptr) return;

  if (config_number == 1)
  {
    ec->interactions_ = &(data.interactions_1);
    // std::cerr << config_number << "int:" << ec->interactions <<"s"<< ec->interactions->size() << std::endl;
  }
  else if (config_number == 0)
  {
    ec->interactions_ = &(data.empty_interactions);
    // std::cerr << config_number << "int:" << ec->interactions <<"s"<< ec->interactions->size() << std::endl;
  }
}

void restore_interactions(tr_data& data, example* ec) { ec->interactions_ = data.backup; }

// for debugging purposes
void print_interactions(example* ec)
{
  if (ec == nullptr) return;
  if (ec->interactions_ == nullptr) return;

  std::cerr << "p:";  // << ec->interactions_;

  for (std::vector<namespace_index> v : *(ec->interactions_))
  {
    for (namespace_index c : v) { std::cerr << " interaction:" << c << ","; }
  }
  std::cerr << std::endl;
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

void print_all_preds(example& ex, size_t i)
{
  const auto& preds = ex.pred.a_s;
  std::cerr << "config_" << i << ": ";
  for (uint32_t i = 0; i < preds.size(); i++)
  {
    std::cerr << preds[i].action << "(" << preds[i].score << ")"
              << ", ";
  }
  std::cerr << std::endl;
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

template <bool is_learn, bool is_explore, typename T>
void predict_or_learn_m(tr_data& data, T& base, multi_ex& ec)
{
  // assert we learn twice
  assert(data.pm == 2);

  if (is_learn) { data.county++; }
  // assert(data.county <= 2000);
  // extra assert just bc
  assert(data.all->_interactions.empty() == true);

  // we force parser to set always as nullptr, see change in parser.cc
  assert(ec[0]->interactions_ == nullptr);
  // that way we can modify all.interactions without parser caring

  CB::cb_class logged{};
  uint32_t labelled_action = 0;
  if (is_learn)
  {
    // we are above shared_feature_merger boohoo?
    // see std::next()
    const auto it =
        std::find_if(std::next(ec.begin()), ec.end(), [](example* item) { return !item->l.cb.costs.empty(); });

    if (it != ec.end())
    {
      logged = (*it)->l.cb.costs[0];
      labelled_action = static_cast<uint32_t>(std::distance(ec.begin(), it));
      // if (labelled_action != 0) std::cerr<<labelled_action<<std::endl;
    }
  }

  // test this works if interactions turns out to be nullptr
  for (uint32_t i = 0; i < data.pm; i++)
  {
    // assert(ec[0]->interactions_ != nullptr);
    assert(data.backup == nullptr);

    for (example* ex : ec) { configure_interactions(data, ex, i); }

    // assert that the config is set correctly
    if (i == 1) { assert(ec[0]->interactions_->empty() != true); }
    else if (i == 0)
    {
      assert(ec[0]->interactions_->empty() == true);
    }

    auto restore_guard = VW::scope_exit([&data, &ec, &i] {
      assert(data.backup == nullptr);
      for (example* ex : ec) { restore_interactions(data, ex); }
      data.backup = nullptr;
    });

    if (!base.learn_returns_prediction || !is_learn) { base.predict(ec, i); }

    // if (i!=1)
    // { // this fixes the lower bound bug
    if (is_learn) { base.learn(ec, i); }
    // }

    if (is_learn)
    {
      const auto action_scores = ec[0]->pred.a_s;

      // NOT FOR NOW cb_explore_adf => want maximum probability
      // cb_adf => first action is a greedy action

      const auto maxit = action_scores.begin();
      const uint32_t chosen_action = maxit->action;

      const float w = logged.probability > 0 ? 1 / logged.probability : 0;
      const float r = -logged.cost;

      if (i == 0) { 
        data.chisq_1.update(chosen_action == labelled_action ? w : 0, r); 
        data.ipsone += r * (chosen_action == labelled_action ? w : 0);
        data.w1 = chosen_action==labelled_action ? w : 0;
        data.r1 = r;
      }
      else if (i == 1)
      {
        data.chisq_2.update(chosen_action == labelled_action ? w : 0, r);
        data.ipstwo += r * (chosen_action == labelled_action ? w : 0);
        data.w2 = chosen_action==labelled_action ? w : 0;
        data.r2 = r;
      }
    }

    // cache the first prediction, if we need to return it (it will get replaced by the second run)
    if (data.which_to_return == 0 && i == 0) { data.a_s = std::move(ec[0]->pred.a_s); }

    // temp print line as if it were finish_example
    // data.adf_learner->print_example(*(data.all), ec);
    // std::cerr << std::endl;
  }

  if (is_learn && data.county % 500 == 0)
  {
    std::cerr << "empty_0:" << data.chisq_1.recompute_duals().first << std::endl;
    std::cerr << "interac_1:" << data.chisq_2.recompute_duals().first << std::endl;
    std::cerr << "ips_0:" << data.ipsone/data.county << std::endl;
    std::cerr << "ips_1:" << data.ipstwo/data.county << std::endl;
    std::cerr << data.county << std::endl << std::endl;
  }

  // replace with prediction depending on which_to_return
  if (data.which_to_return == 0) { ec[0]->pred.a_s = std::move(data.a_s); }

  // assert again just like at the top
  assert(data.all->_interactions.empty() == true);
  assert(ec[0]->interactions_ == nullptr);
}

void persist(tr_data& data, metric_sink& metrics)
{
  metrics.float_metrics_list.emplace_back("test_bound_firstm", data.chisq_1.recompute_duals().first);
  metrics.float_metrics_list.emplace_back("test_bound_secondm", data.chisq_2.recompute_duals().first);
  metrics.float_metrics_list.emplace_back("test_w1", data.w1);
  metrics.float_metrics_list.emplace_back("test_r1", data.r1);
  metrics.float_metrics_list.emplace_back("test_w2", data.w2);
  metrics.float_metrics_list.emplace_back("test_r2", data.r2);
}

void _finish_example(vw& all, tr_data&, multi_ex& ec) { VW::finish_example(all, ec); }

// fail if incompatible reductions got setup
// inefficient, address later
// references global all interactions
void fail_if_enabled(vw& all, std::string name)
{
  if (std::find(all.enabled_reductions.begin(), all.enabled_reductions.end(), name) != all.enabled_reductions.end())
    THROW("plz no bad stack" + name);
}

VW::LEARNER::base_learner* test_red_setup(options_i& options, vw& all)
{
  size_t test_red;
  auto data = scoped_calloc_or_throw<tr_data>();

  option_group_definition new_options("Debug: test reduction");
  new_options.add(make_option("test_red", test_red).necessary().help("blah blah"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  // all is not needed but good to have for testing purposes
  data->all = &all;

  data->which_to_return = test_red;

  // override and clear all the global interactions
  // see parser.cc line 740
  all._interactions.clear();
  assert(all._interactions.empty() == true);

  // make sure we setup the rest of the stack with cleared interactions
  // to make sure there are not subtle bugs
  auto* base_learner = setup_base(options, all);

  // hard code test 312 from RunTests
  // if we comment this following line it works ?? why ??
  // add_interaction(all._interactions, 'G', 'T');
  assert(all._interactions.empty() == true);

  // useful for switching the order around - what params learn'ed first
  add_interaction(data->interactions_1, 'G', 'T');
  // add_interaction(data->empty_interactions, 'G', 'T');

  // ask jack about flushing the cache, after mutating reductions
  // that might change

  fail_if_enabled(all, "ccb_explore_adf");
  fail_if_enabled(all, "audit_regressor");
  fail_if_enabled(all, "baseline");
  fail_if_enabled(all, "cb_explore_adf_rnd");
  fail_if_enabled(all, "cb_to_cb_adf");
  fail_if_enabled(all, "cbify");
  fail_if_enabled(all, "replay_c");
  fail_if_enabled(all, "replay_b");
  fail_if_enabled(all, "replay_m");
  // fail_if_enabled(all, "gd");
  // fail_if_enabled(all, "generate_interactions");
  fail_if_enabled(all, "memory_tree");
  fail_if_enabled(all, "new_mf");
  fail_if_enabled(all, "nn");
  fail_if_enabled(all, "stage_poly");

  // only this has been tested
  if (base_learner->is_multiline)
  {
    // fetch cb_explore_adf to call directly into the print routine twice
    data->adf_learner = as_multiline(base_learner->get_learner_by_name_prefix("cb_explore_adf_"));

    // problem multiplier is set to data->pm
    learner<tr_data, multi_ex>* l = &init_learner(data, as_multiline(base_learner),
        predict_or_learn_m<true, true, multi_learner>, predict_or_learn_m<false, true, multi_learner>, data->pm,
        base_learner->pred_type, all.get_setupfn_name(test_red_setup), true);
    l->set_persist_metrics(persist);
    l->set_finish_example(_finish_example);
    return make_base(*l);
  }
  // not implemented yet
  else
  {
    // problem multiplier is set to data->pm
    learner<tr_data, example>* l = &init_learner(data, as_singleline(base_learner),
        predict_or_learn<true, single_learner>, predict_or_learn<false, single_learner>, data->pm,
        base_learner->pred_type, all.get_setupfn_name(test_red_setup), true);
    l->set_persist_metrics(persist);
    return make_base(*l);
  }
}

}  // namespace test_red
}  // namespace VW
