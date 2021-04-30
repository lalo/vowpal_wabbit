import sys, os
import math
import numpy as np
from scipy.sparse import coo_matrix

from vowpalwabbit import pyvw

class NoopPythonReduction(pyvw.Copperhead):
    # dont do anything here
    def __init__(self):
        super(NoopPythonReduction, self).__init__()
        # create sparse

    def reduction_init(self, vw):
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

    def get_x(self, ec):
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
        
        # sparsej = []
        # sparsev = []

        # for ns_id in range(ec.num_namespaces()):
        #     ord_ns = ec.namespace(ns_id)
        #     #ns = chr(ord_ns)
        #     names = pyvw.namespace_id(ec, ns_id)
        #     #print(names)
        #     for i in range(ec.num_features_in(names.ord_ns)):
        #         # strides per weight
        #         # offset
        #         f = ec.feature(names.ord_ns, i)
        #         # sanity check that we never cross that value
        #         #print(f'currfeathash {f}')
        #         # have to apply some mask second 1 is stride shift
        #         # when you access the weight array - check file array_parameters.h and also dense
        #         # check weight mask
        #         # i can leave this out, just mask with highest num allowed aka num_features
        #         #testy = f & (((1 << self.num_bits) << 1) - 1)
        #         testy = f & (((1 << self.num_bits) << 0) - 1)
        #         #print(testy)
        #         f = testy
        #         assert(f < (self.num_features))
        #         w = ec.feature_weight(names.ord_ns, i)
        #         sparsej.append(f)
        #         sparsev.append(w)

        # X = coo_matrix((sparsev, ([0]*len(sparsej), sparsej)), 
        #                shape=(1, self.num_features),
        #                dtype=np.float32)
        X = self.get_x(ec)
        pred = self.classifier.predict_log_proba(X)

        ec.set_partial_prediction(pred[0, -1] - np.log(0.5))
        #print(f'partial {ec.get_partial_prediction()}')
        # ec.partial_prediction = pred[0, -1] - np.log(0.5)
        # and this is ec.pred.scalar
        ec.set_simplelabel_prediction(pred[0, -1] - np.log(0.5))
        #print(f'simple {ec.get_simplelabel_prediction()}')
        #print(pred[0,-1])
        #y = np.array(["label"])

        #learner.predict(ec)

        # print("hello there I'm predicting stuff")

        # for ns_id in range(ec.num_namespaces()):
        #     ord_ns = ec.namespace(ns_id)
        #     #ns = chr(ord_ns)
        #     names = pyvw.namespace_id(ec, ns_id)
        #     #print(names)
        #     for i in range(ec.num_features_in(names.ord_ns)):
        #         f = ec.feature(names.ord_ns, i)
        #         w = ec.feature_weight(names.ord_ns, i)
        #         print(f)
        #         print(w)

        #learner.predict(ec)

    def _learn(self, ec, learner):
        X = self.get_x(ec)

        # check before if its simplelabel
        label = ec.get_simplelabel_label()

        Y = np.array([ 1 if label > 0 else 0 ])

        self.classifier.partial_fit(X, Y)

        #learner.learn(ec)
        # print("hello there I can also learn stuff btw the total_sum_feat_sq is " + str(ec.get_total_sum_feat_sq()))
        # for ns_id in range(ec.num_namespaces()):
        #     ord_ns = ec.namespace(ns_id)
        #     ns = chr(ord_ns)
        #     print(ns)
        #     for i in range(ec.num_features_in(ns)):
        #         print(i)
        #learner.learn(ec)

# 1) need to know the shape of feature matrix
# 2) the class must have an init function that takes in the shape
# 3) DRY
# - vw specifies size via bits, can pull that setting from *all
# binary reduction -> straightforward -> directly related to number of bits (might not apply)

# nico and sebastian

# transform ec -> raw features -> populate sparse np matrix
# you have to say the shape of the matrix, when you construct shape=(1, n_features) ==>> 1 example, need vw to tell me n_features
# i have a sparse matrix -> feed directly to predict and partial fit for incremental algos
# unmarshal result and pump back to vw "ec"

# this is a recreation of the impl in vw
# see https://github.com/VowpalWabbit/vowpal_wabbit/blob/ac3a2c21a9760b68ce49368b11a35bf95faeb8b8/vowpalwabbit/binary.cc
class BinaryPythonReduction(pyvw.Copperhead):
    def reduction_init(self, vw):
        print("hola")

    def _predict(self, ec, learner):
        # print("hello there I'm predicting stuff")
        learner.predict(ec)

    def _learn(self, ec, learner):
        learner.learn(ec)

        if ec.get_simplelabel_prediction() > 0:
            ec.set_simplelabel_prediction(1)
        else:
            ec.set_simplelabel_prediction(-1)

        temp_label = ec.get_simplelabel_label()

        # going to have to expose FLT_MAX?
        if temp_label != sys.float_info.max:
            if math.fabs(temp_label) != 1.0:
                print("You are using label " + temp_label + " not -1 or 1 as loss function expects!")
            elif temp_label == ec.get_scalar():
                ec.set_loss(0)
            else:
                ec.set_loss(ec.get_simplelabel_weight())



print(os.getpid())
#scikitlearn first

if sys.version_info > (3, 0):
    print("good, python3")
else:
    raise Exception("you are not on python 3")


#pyvw.get_config(gd, scorer, cost_sensitive, cb)



# this should match cpp_binary() output
# doesn't do anything, runs in python see class impl NoopPythonicReductions
def noop_example():
    #vw = pyvw.vw(python_reduction=NoopPythonReduction, arg_str="--loss_function logistic --binary --active_cover --oracular -d /root/vw/test/train-sets/rcv1_small.dat")
    vw = pyvw.vw(python_reduction=NoopPythonReduction, arg_str="--loss_function logistic --binary  -d /root/vw/test/train-sets/rcv1_small.dat")
    #print(vw.get_stride())

    #vw.run_parser()
    #vw.finish()

    #prediction = vw.predict("-1 |f 9:6.2699720e-02 14:3.3754818e-02")
    #vw.learn("-1 |f 9:6.2699720e-02 14:3.3754818e-02")
    #print(prediction)
    #vw.run_parser()
    config = vw.get_config()

    # print("hola " + str(config["active_cover"][0][1][0]))

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

    

    vw.finish()

# this should match cpp_binary() output
# mirror implementation of the cpp, runs in python see class impl BinaryPythonReductions
def python_binary():
    vw = pyvw.vw(enable_logging=True, python_reduction=BinaryPythonReduction, arg_str="--loss_function logistic --active_cover --oracular -d /root/vowpal_wabbit/test/train-sets/rcv1_small.dat")
    vw.run_parser()
    vw.finish()
    return vw.get_log()

# this should be the baseline
def cpp_binary():
    vw = pyvw.vw("--loss_function logistic --binary --active_cover --oracular -d /root/vowpal_wabbit/test/train-sets/rcv1_small.dat", enable_logging=True)
    vw.run_parser()
    vw.finish()
    return vw.get_log()

# print("python")
# python_binary()
# print("noop")
# noop_example()
# print("cpp")
# cpp_binary()

import pytest
import os
print(os.getpid())

def test_python_binary_reduction():
    python_binary_log = python_binary()
    native_binary_log = cpp_binary()
    assert len(python_binary_log) == len(native_binary_log)

    line_number = 0
    for j, i in zip(python_binary_log, native_binary_log):
        if line_number == 7:
            assert "Enabled reductions" in j
            assert "Enabled reductions" in i
            line_number += 1
            continue

        assert i == j, "line mismatch should be: " + j + " output: " + i
        line_number += 1
