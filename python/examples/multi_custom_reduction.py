import sys, os

from vowpalwabbit import pyvw

class MultiReduction(pyvw.Copperhead):
    def __init__(self):
        super(MultiReduction, self).__init__()

    def reduction_init(self, vw):
        print("reduction init post vw init")

    def _predict(self, examples, learner):
        print("predicting")

    def _learn(self, examples, learner):
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