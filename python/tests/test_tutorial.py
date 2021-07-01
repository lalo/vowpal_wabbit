from vowpalwabbit import pyvw

import pytest
import random

# this test was adapted from this tutorial: https://vowpalwabbit.org/tutorials/cb_simulation.html

# VW tries to minimize loss/cost, therefore we will pass cost as -reward
USER_LIKED_ARTICLE = -1.0
USER_DISLIKED_ARTICLE = 0.0

curr_file = None

def get_cost(context,action):
    if context['user'] == "Tom":
        if context['time_of_day'] == "morning" and action == 'politics':
            return USER_LIKED_ARTICLE
        elif context['time_of_day'] == "afternoon" and action == 'music':
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE
    elif context['user'] == "Anna":
        if context['time_of_day'] == "morning" and action == 'sports':
            return USER_LIKED_ARTICLE
        elif context['time_of_day'] == "afternoon" and action == 'politics':
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE

# This function modifies (context, action, cost, probability) to VW friendly format
def to_vw_example_format(context, actions, cb_label = None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared |Gser user={} time_of_day={}\n".format(context["user"], context["time_of_day"])
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += "0:{}:{} ".format(cost, prob)
        example_string += "|Tction article={} \n".format(action)
    #Strip the last newline
    return example_string[:-1]


def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1 / total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    print("draw="+str(draw), file=curr_file)
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if(sum_prob > draw):
            print("sum_prob="+str(sum_prob), file=curr_file)
            return index, prob

def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context,actions)
    pmf = vw.predict(vw_text_example)
    print("pmf:" +str(pmf), file=curr_file)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    print(str(chosen_action_index)+" "+str(prob), file=curr_file)
    return actions[chosen_action_index], prob

users = ['Tom', 'Anna']
times_of_day = ['morning', 'afternoon']
actions = ["politics", "sports", "music", "food", "finance", "health", "camping"]

def choose_user(users):
    return random.choice(users)

def choose_time_of_day(times_of_day):
    return random.choice(times_of_day)

def run_simulation(vw, num_iterations, users, times_of_day, actions, cost_function, do_learn = True):
    cost_sum = 0.
    ctr = []

    for i in range(1, num_iterations+1):
        print("", file=curr_file)
        print("id:"+str(i),file=curr_file)
        # 1. In each simulation choose a user
        user = choose_user(users)
        # 2. Choose time of day for a given user
        time_of_day = choose_time_of_day(times_of_day)

        # 3. Pass context to vw to get an action
        context = {'user': user, 'time_of_day': time_of_day}
        action, prob = get_action(vw, context, actions)

        # 4. Get cost of the action we chose
        cost = cost_function(context, action)
        print("cost="+str(cost), file=curr_file)
        cost_sum += cost

        if do_learn:
            # 5. Inform VW of what happened so we can learn from it
            # if (cost == 0):
            #     print(actions.index(action))
            vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)),pyvw.vw.lContextualBandit)
            # 6. Learn
            vw.learn(vw_format)
            # 7. Let VW know you're done with these objects
            vw.finish_example(vw_format)

        # We negate this so that on the plot instead of minimizing cost, we are maximizing reward
        ctr.append(-1*cost_sum/i)

    return ctr

def test_with_interaction():
    vw = pyvw.vw("--random_seed 5 --cb_explore_adf -q GT --quiet --epsilon 0.2")
    num_iterations = 2000
    random.seed(10)
    global curr_file
    curr_file = open('with_inter.txt', 'w')
    ctr = run_simulation(vw, num_iterations, users, times_of_day, actions, get_cost)
    curr_file.close()

    print("with interaction")
    print(ctr[-1])
    assert(ctr[-1] == 0.765)

def test_without_interaction():
    vw = pyvw.vw("--random_seed 5 --cb_explore_adf --quiet --epsilon 0.2")
    num_iterations = 2000
    random.seed(10)
    global curr_file
    curr_file = open('without_inter.txt', 'w')
    ctr = run_simulation(vw, num_iterations, users, times_of_day, actions, get_cost)
    curr_file.close()

    print("without interaction")
    print(ctr[-1])
    assert(ctr[-1] == 0.4035)

def test_custom_reduction(config=0):
    # set test_red to 1 to return pred of with interaction
    # set test_red to 0 to return pred of no interaction
    vw = pyvw.vw("--random_seed 5 --test_red "+ str(config) +" --cb_explore_adf --quiet --epsilon 0.2")
    num_iterations = 2000
    random.seed(10)
    global curr_file
    curr_file = open("custom_reduc_"+str(config)+".txt", 'w')
    ctr = run_simulation(vw, num_iterations, users, times_of_day, actions, get_cost)
    curr_file.close()

    print("custom reduction - "+str(config))
    print(ctr[-1])
    if config == 0:
        assert(ctr[-1] == 0.371)
    elif config == 1:
        assert(ctr[-1] == 0.766)
    else:
        assert(false)

test_with_interaction()
test_without_interaction()

with_interaction = 1
without_interaction = 0
test_custom_reduction(with_interaction)
test_custom_reduction(without_interaction)
