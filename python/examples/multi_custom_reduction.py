import sys, os

from vowpalwabbit import pyvw

class MultiReduction(pyvw.Copperhead):
    def __init__(self):
        super(MultiReduction, self).__init__()

    def reduction_init(self, vw):
        print("reduction init post vw init")

    def _predict(self, examples, learner):
        #print("predicting as usual / identity")
        learner.multi_predict(examples)

    # during predict you are the identity reduction
    # during learn you do something else:
    # run on top of cb_adf
    # call base predict, you get a best action ("first action returned")
    # logged probability
    # w = importance weight equals:
        # 0 if logged action != best action
        # 1/(logged probability) if logged action == best action
    # r = reward is also in the log
    # maintain a list of historical (w, r) pairs
    def _learn(self, examples, learner):
        learner.multi_predict(examples)

        example_with_label, labelled_action = examples.get_example_with_label()

        if labelled_action != -1:
            print(labelled_action)

            logged_cost = example_with_label.get_cbandits_cost(0)
            logged_prob = example_with_label.get_cbandits_probability(0)

            action_scores = examples[0].get_action_scores()

            # using cb_adf: first action is best
            chosen_action = action_scores[0]

            if chosen_action != labelled_action:
                w = 0
            else:
                w = 1 / logged_prob
            r = -logged_cost

            print(f'w {w}')
            print(f'r {r}')

            # TODO mantain list 


        # learner.multi_learn(examples, ..)
        print("learning")

def sanity_check():
    # useful for attaching gbd debugger
    print(os.getpid())

    # not necessary but python 2 is on its way out
    if sys.version_info > (3, 0):
        print("good, python3")
    else:
        raise Exception("you are not on python 3")

def print_config(config):
    cmd_str = []

    for name, config_group in config.items():
        print(f'Reduction name: {name}')
        for (group_name, options) in config_group:
            print(f'\tOption group name: {group_name}')

            #if name == "general":
            #    continue

            for option in options:
                print(f'\t\tOption name: {option.name}, keep {option.keep}, help: {option.help_str}')
                temp_str = str(option)
                if temp_str:
                    cmd_str.append(temp_str)

    print(cmd_str)


def run_example():
    vw = pyvw.vw(python_reduction=MultiReduction, arg_str=" --cb_adf --rank_all -d /root/vowpal_wabbit/test/train-sets/cb_adf_sm.data --cb_type sm")

    vw.run_parser()
    config = vw.get_config()

    vw.finish()
    #print_config(config)

sanity_check()
print("Running...")
run_example()