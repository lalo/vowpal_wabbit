import sys, os
import math
import numpy as np
import torch

from vowpalwabbit import pyvw

class PyTorchReduction(pyvw.Copperhead):
    def __init__(self):
        super(PyTorchReduction, self).__init__()
        # create sparse

    def reduction_init(self, vw):
        # get initial l rate
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=...) 

        from sklearn.linear_model import SGDClassifier

        config = vw.get_config()
        # 1. set of all classes (0 and 1 for binary -> labels)
        self.classes = np.array([0,1])

        self.num_bits = config["general"][6][1][6].value
        self.num_bits = 18
        #self.num_features = 1 << self.num_bits

        # weight length = (1 << 18) << ss
        self.num_features = 1 << 18

        self.classifier = SGDClassifier(loss='log')
        # tuple, first element are the values the second is another tuple
        # first is row index, then colx
        fakeX = coo_matrix(([], ([], [])), shape=(1, self.num_features))
        fakeY = np.array([0])
        self.classifier.partial_fit(fakeX, fakeY, classes=self.classes)

    def vw_ex_to_scikit(self, ec):
        sparsej = []
        sparsev = []

        for ns_id in range(ec.num_namespaces()):
            # 128 ord_ns is constant namespace
            ord_ns = ec.namespace(ns_id)
            names = pyvw.namespace_id(ec, ns_id)
            for i in range(ec.num_features_in(names.ord_ns)):
                f = ec.feature(names.ord_ns, i)
                f = f & (((1 << self.num_bits) << 0) - 1)

                # sanity check
                assert(f < (self.num_features))

                w = ec.feature_weight(names.ord_ns, i)
                sparsej.append(f)
                sparsev.append(w)

        X = coo_matrix((sparsev, ([0]*len(sparsej), sparsej)), 
                       shape=(1, self.num_features),
                       dtype=np.float32)
        
        return X

    def _predict(self, ec, learner):
        with torch.no_grad():
            pred = self.model.forward(self.vw_ex_to_pytorch(ec))

        ec.set_partial_prediction(pred[0, -1] - np.log(0.5))
        ec.set_simplelabel_prediction(pred[0, -1] - np.log(0.5))

    def _learn(self, ec, learner):
        self.optimizer.zero_grad()
        pred = self.model.forward(self.vw_ex_to_pytorch(ec))
        loss = self.vw_ex_to_weight(ec) * self.lossfn(pred, self.vw_ex_to_label(ec))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

print(os.getpid())

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

# this should match cpp_binary() output
# doesn't do anything, runs in python see class impl NoopPythonicReductions
def noop_example():
    vw = pyvw.vw(python_reduction=PyTorchReduction, arg_str="--loss_function logistic --binary  -d /root/vw/test/train-sets/rcv1_small.dat")
    #print(vw.get_stride())
    vw.run_parser()

    print_config(vw.get_config())
    vw.finish()
    #prediction = vw.predict("-1 |f 9:6.2699720e-02 14:3.3754818e-02")
    #vw.learn("-1 |f 9:6.2699720e-02 14:3.3754818e-02")

print("noop")
noop_example()