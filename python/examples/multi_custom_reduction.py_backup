import sys, os

from vowpalwabbit import pyvw

class MultiReduction(pyvw.Copperhead):
    def __init__(self):
        super(MultiReduction, self).__init__()
        self.allwrs = []
        self.wmax = 2

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
            # r = -logged_cost
            r = 1.5 - logged_cost

            print(f'w {w}')
            print(f'r {r}')

            # TODO: this will get butt slow after a while
            # the fix is to approximate the history with a histogram of ((w, r), c) i.e. a count for each (w, r) pair
            # you will need to discretize (w, r) at some resolution
            # then you need to adjust the routine islowerbound() so that all the sums over the dataset are implemented as count-weighted sums

            # TIP #1: rewards in [0, 1]? then use fractional counts at r=0 and r=1
            # (w, 0), c
            # (w, 1), c
            # TIP #2: discretize on log(1+w) rather than w
            #         np.log(np.geomspace(1, wmax + 1, 50)) => array of 50 points in log(1+w) space

            self.allwrs.append((w, r))
            self.wmax = max(self.wmax, w)
            if w > 0 and len(self.allwrs) > 1:
                lb = MultiReduction.islowerbound(self.allwrs, 0, self.wmax)
                exweight = max(0.01, len(self.allwrs) * lb['pstarfunc'](w, r) / w)
            else:
                exweight = 1

            print(f'exweight {exweight}')

            # TODO: 
            # 1. scale example weight by exweight
            # 2. pass down reduction stack for learning
            # 3. restore original example weight


        # learner.multi_learn(examples, ..)
        print("learning")

    @staticmethod
    def islowerbound(wrs, wmin, wmax, coverage=0.9):
        import numpy as np

        assert 0 <= wmin < 1
        assert wmax > 1

        np.seterr(invalid='raise')

        # massage into IS form

        dataset = lambda: wrs
        N = len(wrs)

        # solve MLE
        if True:
            from scipy import optimize
            assert all(wmin <= w <= wmax for w, _ in dataset()), (min(w for w, _ in dataset()), max(w for w, _ in dataset()))
            def dual(beta):
                from math import fsum
                # example histogram version
                #return -fsum((c/N) * np.log(1 + beta * (w - 1)) for (w, r), c in dataset())
                return -fsum((1/N) * np.log(1 + beta * (w - 1)) for w, _ in dataset())
            betamin = 1e-6 - 1/(wmax - 1)
            betamax = -1e-6 + 1/(1 - wmin)
            res = optimize.minimize_scalar(fun=dual, method='bounded', bounds=(betamin, betamax))
            assert res.success, ('MLE', res)
            phiminusallentropy = res.fun
            betamle = res.x

        # solve bound
        if True:    
            from scipy.stats import f
            from cvxopt import solvers, matrix

            assert all(0 <= r for _, r in dataset()), (min(r for _, r in dataset()), max(r for _, r in dataset()))

            Delta = 0.5 * f.isf(q=1.0-coverage, dfn=1, dfd=N-1) / N
            phi = phiminusallentropy - Delta

            x0 = np.array([ 1.0, 1.0 - betamle, betamle ])

            def negdualobjective(p):
                from math import fsum
                from scipy import special 

                (kappa, alpha, beta) = p[0], p[1], p[2]

                return - ( fsum( (1/N) * w * r for w, r in dataset() ) + 
                           beta * (-1 + sum((1/N) * w for w, r in dataset())) + 
                           kappa * phi
                           - fsum( (1/N) * special.kl_div(kappa, alpha + beta * w + w * r) for w, r in dataset() )
                         )

            def dkldivdx(x, y):
                # x log(x/y) - x + y
                return np.log(x / y)

            def dkldivdy(x, y):
                return 1 - x / y

            def jacnegdualobjective(p):
                from math import fsum

                (kappa, alpha, beta) = p[0], p[1], p[2]
                jacobj = np.empty(3)

                jacobj[0] = -phi + fsum( (1/N) * dkldivdx(kappa, alpha + beta * w + w * r) for w, r in dataset() )
                jacobj[1] = fsum( (1/N) * dkldivdy(kappa, alpha + beta * w + w * r) for w, r in dataset() )
                jacobj[2] = -(-1 + fsum( (1/N) * w for w, r in dataset() )) + fsum( (1/N) * w * dkldivdy(kappa, alpha + beta * w + w * r) for w, r in dataset() )

                return jacobj

            def d2kldivdx2(x, y):
                return 1 / x

            def d2kldivdxdy(x, y):
                return -1 / y

            def d2kldivdy2(x, y):
                return x / y**2

            def hessnegdualobjective(p):
                from math import fsum

                (kappa, alpha, beta) = p[0], p[1], p[2]
                hessobj = np.empty((3, 3))

                hessobj[0][0] = fsum( (1/N) * d2kldivdx2(kappa, alpha + beta * w + w * r) for w, r in dataset() )
                hessobj[0][1] = fsum( (1/N) * d2kldivdxdy(kappa, alpha + beta * w + w * r) for w, r in dataset() )
                hessobj[0][2] = fsum( (1/N) * w * d2kldivdxdy(kappa, alpha + beta * w + w * r) for w, r in dataset() )
                hessobj[1][0] = hessobj[0][1]
                hessobj[1][1] = fsum( (1/N) * d2kldivdy2(kappa, alpha + beta * w + w * r) for w, r in dataset() )
                hessobj[1][2] = fsum( (1/N) * w * d2kldivdy2(kappa, alpha + beta * w + w * r) for w, r in dataset() )
                hessobj[2][0] = hessobj[0][2]
                hessobj[2][1] = hessobj[1][2]
                hessobj[2][2] = fsum( (1/N) * w**2 * d2kldivdy2(kappa, alpha + beta * w + w * r) for w, r in dataset() )

                return hessobj

            tiny = 1e-6
            def F(x=None, z=None):
                if x is None: return 0, matrix(x0)

                kappa, alpha, beta = x[0], x[1], x[2]

                if kappa < tiny or any(alpha + beta * w + w * r < tiny for w, r in dataset()):
                    return None

                f = negdualobjective(x)
                jf = jacnegdualobjective(x)
                Df = matrix(jf).T
                if z is None: return f, Df
                hf = z[0] * hessnegdualobjective(x)
                H = matrix(hf, hf.shape)
                return f, Df, H

            consE = np.array([ [ 1.0, 0.0, 0.0 ],
                               [ 0.0, 1.0, wmin ],
                               [ 0.0, 1.0, wmax ]
                             ])
            d = np.array([ tiny, tiny, tiny ])

            soln = solvers.cp(F,
                              G = -matrix(consE, consE.shape),
                              h = -matrix(d),
                              options = { 'show_progress': False })

            from pprint import pformat
            assert soln['status'] == 'optimal', pformat(soln)
            kappastar, alphastar, betastar = soln['x']
            vhat = -soln['primal objective']

        pstarfunc = lambda w, r: kappastar / (alphastar + betastar * (w - 1) + w * r)

        return { 'vhat': vhat, 'alpha': alphastar, 'beta': betastar, 'kappa': kappastar, 'pstarfunc': pstarfunc }


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
    vw = pyvw.vw(python_reduction=MultiReduction, arg_str=" --cb_adf -d /root/vowpal_wabbit/test/train-sets/cb_adf_sm.data")

# to use with cb_adf: needs contextual bandit dataset (like the kind sent to the trainer in APS ... maybe ask marco)
# with cb_explore_adf, can use a supervised dataset with --cbify
    vw.run_parser()
    config = vw.get_config()

    vw.finish()
    #print_config(config)

sanity_check()
print("Running...")
run_example()