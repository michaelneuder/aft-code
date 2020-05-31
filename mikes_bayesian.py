#!/usr/bin/env python
# coding: utf-8

from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np
import progressbar

# results in this format from the output of the multi-threaded c++ code.
resultsFinal = np.asarray([
0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0,
3e-08, 0, 0, 0, 0, 0,
6e-08, 0, 0, 0, 0, 0,
2.5e-07, 0, 0, 0, 0, 0,
7.1e-07, 0, 0, 0, 0, 0,
1.65e-06, 0, 0, 0, 0, 0,
4.9e-06, 0, 0, 0, 0, 0,
1.178e-05, 0, 0, 0, 0, 0,
2.797e-05, 0, 0, 0, 0, 0,
6.254e-05, 0, 0, 0, 0, 0,
0.00013422, 2e-08, 0, 0, 0, 0,
0.00027849, 7e-08, 0, 0, 0, 0,
0.00054056, 1.5e-07, 0, 0, 0, 0,  
0.00101865, 8.4e-07, 0, 0, 0, 0,
0.00185053, 3.71e-06, 0, 0, 0, 0,
0.00323235, 1.331e-05, 0, 0, 0, 0,
0.0054424, 3.902e-05, 0, 0, 0, 0,
0.00889944, 0.00011233, 1e-08, 0, 0, 0,
0.0140104, 0.00029576, 8e-08, 0, 0, 0,
0.0214406, 0.00073342, 1.19e-06, 0, 0, 0,
0.0318141, 0.00168044, 6.31e-06, 2e-08, 0, 0,
0.0459994, 0.00364552, 3.162e-05, 2e-08, 0, 0,
0.0646293, 0.00743669, 0.0001314, 4.6e-07, 0, 0,
0.0885937, 0.0142396, 0.00049523, 4.14e-06, 2e-08, 0,
0.118352, 0.0257663, 0.00162757, 3.233e-05, 3.6e-07, 0, 
0.154433, 0.0440788, 0.00472106, 0.00020428, 3.66e-06, 1e-08, 
0.196837, 0.0715445, 0.0122646, 0.00105628, 4.37e-05, 9.2e-07, 
0.245502, 0.110379, 0.0283845, 0.00443653, 0.00041492, 2.269e-05, 
0.299761, 0.162142, 0.0591246, 0.0153717, 0.00283166, 0.00037416, 
0.358843, 0.227416, 0.111237, 0.0440746, 0.0141348, 0.00365242, 
0.421386, 0.305263, 0.189755, 0.105458, 0.0524408,  0.0232268,
0.485932, 0.392829, 0.295232, 0.212378, 0.146406, 0.0962258, 
0.550785, 0.486158, 0.421551, 0.364836, 0.314247, 0.268755,])
resultsFinal = np.reshape(resultsFinal, (40,6))


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
    bar = progressbar.ProgressBar()
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


length_20_probs = resultsFinal[:,2]
length_20_probs

length_20_probs_nonzero = length_20_probs[24:]
length_20_probs_nonzero


length_20_probs_nonzero_geq105 = length_20_probs[28:]
length_20_probs_nonzero_geq105

alphas_geq105 = np.arange(0.38, 0.50, 0.01)
alphas_geq105

assert(len(alphas_geq105) == len(length_20_probs_nonzero_geq105))

def objective(inputs):
    print(inputs)
    val = 0
    bar = progressbar.ProgressBar()
    for i in bar(range(12)):
        prob = getProbReorg(alpha = alphas_geq105[i], 
                            length=20, 
                            init_endorsers = inputs[0][0], 
                            delay_priority = inputs[0][1], 
                            delay_endorse = inputs[0][2],
                            sample_size=int(1e5))
        val += prob / length_20_probs_nonzero_geq105[i]
    print("value: ", val)
    return val

domain = [{'name': 'init_endorsers', 'type': 'discrete', 'domain': tuple(range(33))}, 
        {'name': 'delay_priority', 'type': 'discrete', 'domain': tuple(range(100))}, 
        {'name': 'delay_endorse', 'type': 'discrete', 'domain': tuple(range(100))}]


opt = BayesianOptimization(f = objective, domain = domain)
opt.run_optimization(max_iter = 100)
opt.plot_acquisition()

