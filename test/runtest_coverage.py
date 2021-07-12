from vowpalwabbit import pyvw

def get_latest_tests():
    import runtests_parser as rtp
    tests = rtp.file_to_obj(rtp.find_runtest_file())
    return [x.__dict__ for x in tests]


def get_all_options():
    return pyvw.get_all_vw_options()


def to_json():
    config = get_all_options()
    for name, config_group in config.items():
        for (group_name, options) in config_group:
            for option in options:
                option._type = str(type(option._default_value).__name__)
    
    import json

    with open("vw_options.json", "w") as f:
        f.write(json.dumps(config, indent=2, default=lambda x: x.__dict__))


def print_option_group(desc, options):
    if options:
        print(desc)
        for l in options:
            print("\t\t// "+str(l._seq_id)+". "+l.help_str)
            if l.default_value_supplied:
                if l._type == "str":
                    print("\t\t"+l._type+" " +l.name+" = \""+str(l.default_value)+"\";")
                else:
                    print("\t\t"+l._type+" " +l.name+" = "+str(l.default_value)+";")
            else:
                print("\t\t"+l._type+" " +l.name+";")
        print()


# dummy codegen
# prints constructor for option group
def necessary_to_constructor(name, options):
    if len(options) == 0:
        print("\t"+name + "() {}")
    elif len(options) == 1:
        if options[0]._type == "bool":
            print("\t"+name + "() {}")
        else:
            print("\t"+name + "("+options[0]._type+" "+options[0].name+") {}")
    elif len(options) == 2:
        if "cb_explore_adf_" in name:
            if "bool" == options[1]._type:
                print("\t"+name + "() {}")
            else:
                print("\t"+name + "("+options[1]._type+" "+options[1].name+") {}")
        else:
            raise("not possible")
    else:
        raise("not possible")
    print()


def print_dummy_contract():
    print("\t// fn to be used for add_parse_and_check_necessary of")
    print("\t// config_backed_options (which would implement options_i)")
    print("\t// this method is part of config_i interface")
    print("\tbool process_option_group(option_group_definition ogd);")
    print()


def print_group(name, group_name, group_id, options):
    print("// "+ str(group_id) + ". " + group_name)
    print("// total options: " + str(len(options)))
    print("pseudoclass " + name + "_config : config_i {")

    necessary = list(filter(lambda x: x.necessary, options)) 
    keep = list(filter(lambda x: x.keep, options)) 
    bools = list(filter(lambda x: x._type == "bool" and x.keep == False, options)) 
    default = list(filter(lambda x: x.default_value_supplied and x._type != "bool", options)) 
    default = list(filter(lambda x: not x.keep and not x.necessary, default)) 
    rest = list(filter(lambda x: not x.default_value_supplied and x._type != "bool", options)) 
    rest = list(filter(lambda x: not x.keep and not x.necessary, rest)) 
        
    print_option_group("\tNECESSARY (turn on reduction):", necessary)
    necessary_to_constructor(name+"_config", necessary)
    print_dummy_contract()
    print_option_group("\tKEEP (persisted in the model):", keep)
    print_option_group("\tOTHER FLAGS (bools & !keep):", bools)
    print_option_group("\tHAS_DEFAULT:", default)
    print_option_group("\tNO_DEFAULT:", rest)

    print("};\n")


def prettyprint(config, filter_out_general=True):
    group_id = 1
    for name, config_group in config.items():
        # general are the VW generic options
        if filter_out_general and name == "general":
            continue
        # temp fix:
        if name == "cb_adf":
            config_group.pop()
        for (group_name, options) in config_group:
            num_id = 1
            for option in options:
                option._type = str(type(option._default_value).__name__)
                option._seq_id = num_id
                num_id += 1
            print_group(name, group_name, group_id, options)
            group_id += 1


vw = pyvw.vw(arg_str="--cb_explore_adf")
prettyprint(vw.get_config())

# prettyprint(get_all_options(), False)
exit()

def get_config_of_vw_cmd(test):
    vw = pyvw.vw(arg_str=test["vw_command"])
    config = vw.get_config()
    enabled_reductions = vw.get_enabled_reductions()
    vw.finish()
    return config, enabled_reductions


def update_option(config, name, group_name, option_name):
    for (g_n, options) in config[name]:
        if g_n == group_name:
            for option in options:
                if option.name == option_name:
                    option.value = True

    return config


def merge_config(tracker, b):
    for name, config_group in b.items():
        for (group_name, options) in config_group:
            for option in options:
                if option.value_supplied:
                    tracker = update_option(tracker, name, group_name, option.name)

    return tracker


def print_non_supplied(config):
    with_default = []
    without_default = []

    for name, config_group in config.items():
        for (group_name, options) in config_group:
            for option in options:
                if not option.value_supplied:
                    default_val_str = ""
                    if option.default_value_supplied:
                        default_val_str = ", BUT has default value"
                        agg = with_default

                    if len(config_group) <= 1:
                        agg.append(name + ", " + option.name + default_val_str)
                    else:
                        agg.append(name + ", " + group_name + ", " + option.name + default_val_str)
                    
                    agg = without_default
    
    for e in with_default:
        print(e)
    for e in without_default:
        print(e)
        

# this function needs more depedencies (networkx, matplotlib, graphviz)
def draw_graph(stacks):
    import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import write_dot, graphviz_layout

    G=nx.DiGraph()

    for l in stacks:
        reductions = l.split("->")
        for i,k in zip(reductions, reductions[1:]):
            G.add_edge(k.replace("cb_explore_adf_",""),i.replace("cb_explore_adf_",""))
        if len(reductions) == 1:
            G.add_node(reductions[0])

    plt.figure(num=None, figsize=(24, 12), dpi=120, facecolor='w', edgecolor='k')

    write_dot(G,'graphviz_format.dot')

    pos =graphviz_layout(G, prog='dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=12')
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=1600)
    plt.savefig('reduction_graph.png')


def main():
    stacks = []

    allConfig = get_all_options() 
    tests = get_latest_tests()
    for test in tests:
        # fails for unknown reasons (possibly bugs with pyvw)
        if test["id"] in [195, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 258, 269]:
            continue

        if "vw_command" in test:
            config, enabled_reductions = get_config_of_vw_cmd(test)
            stacks.append('->'.join(enabled_reductions))
            allConfig = merge_config(allConfig, config)
    
    print_non_supplied(allConfig)

    # draw_graph(stacks)

    # print reduction stack by count
    from collections import Counter
    for c in Counter(stacks).most_common():
        print(c[0]+", "+str(c[1]))


if __name__ == "__main__":
    main()
