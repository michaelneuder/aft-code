#!/usr/bin/env python
# coding: utf-8

from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np
import progressbar

# honest network delay over next n blocks.
def vectorDelayHonest(ps, es, init_endorsers, delay_priority, delay_endorse):
    return (60 * len(ps)
           + delay_priority * sum(ps) 
           + sum([delay_endorse * max(init_endorsers - e, 0) for e in es]))

# attacking network delay over next n blocks.
def vectorDelayAttacker(ps, es, init_endorsers, delay_priority, delay_endorse):
    return (60 * len(ps) 
           + delay_priority * sum(ps) 
           + sum([delay_endorse * max(init_endorsers - e, 0) for e in es[1:]]))

# efficient sample generation
def getAH(alpha):
    x = np.random.geometric(1-alpha)
    if x == 1:
        h = 0
        a = np.random.geometric(alpha)
    else:
        a = 0
        h = x - 1
    return [a, h]

def getProbReorg(alpha, length, init_endorsers, delay_priority, delay_endorse, sample_size = int(1e5)):
    feasible_count = 0
    for _ in range(sample_size):
        aVals = []
        hVals = []
        for i in range(length):
            a, h = getAH(alpha)
            aVals.append(a)
            hVals.append(h)
        eVals = np.random.binomial(32, alpha, size = length)
        honest_delay = vectorDelayHonest(hVals, 32 - eVals, init_endorsers, delay_priority, delay_endorse)
        selfish_delay = vectorDelayAttacker(aVals, eVals, init_endorsers, delay_priority, delay_endorse)
        if selfish_delay <= honest_delay:
            feasible_count += 1
    return feasible_count / sample_size

for alpha_ in np.arange(0.38, 0.5, 0.01):
    print(alpha_)
    def objective(inputs):  
        prob = getProbReorg(alpha = alpha_, 
                            length=20, 
                            init_endorsers = inputs[0][0], 
                            delay_priority = inputs[0][1], 
                            delay_endorse = inputs[0][2],
                            sample_size=int(1e5))
        return prob

    domain = [{'name': 'init_endorsers', 'type': 'discrete', 'domain': tuple(range(33))}, 
            {'name': 'delay_priority', 'type': 'discrete', 'domain': tuple(range(100))}, 
            {'name': 'delay_endorse', 'type': 'discrete', 'domain': tuple(range(100))}]


    opt = BayesianOptimization(f = objective, domain = domain)
    opt.run_optimization(max_iter = 100)
    print("\nX = [init_endorsers delay_priority delay_endorse]")
    print("\nAll tested values of X : \n" + str(opt.X))
    print("\nCorresponding values of Y : \n" + str(opt.Y))
    print("\nOptimal X : \n" + str(opt.x_opt))
    print("\nOptimal Y : \n" + str(objective([opt.x_opt])))
    print("\nCurrent Tezos : \n" + str(objective([[24, 40, 8]])))

