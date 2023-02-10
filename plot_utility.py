import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import itertools
import sys
sys.path.append("/mnt/archgen/users/yilei/IBD/timeSampling")
from ibdDemo.analytic import singlePop_2tp_given_vecNe

minl = 4
maxl = 20
step = 0.25
bins = np.arange(minl, maxl+step, step)
midpoint = (bins[1:]+bins[:-1])/2
L = np.array([286.279, 268.840, 223.361, 214.688, 204.089, 192.040, 187.221, 168.003, 166.359, \
        181.144, 158.219, 174.679, 125.706, 120.203, 141.860, 134.038, 128.491, 117.709, \
        107.734, 108.267, 62.786, 74.110])

########################### the following function uses gridspec to draw estimated Ne togerhter with IBD fitting ############
########################## it's designed to work with 3 sampling time (thus 6 subpanels for IBD fitting) ####################

def plot(ibds, nSamples, gaps, trueNe, estNe, labelNe, lowCIs, highCIs, outFolder, prefix):
    # estNe: list of list or a single list
    # seNe: list of list or a single list, in the same order as estNe
    # labelNe: label of each Ne list in estNe, in the same order
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(3, 3, width_ratios=[3, 1,1])

    ################## plot estimated Ne ####################
    ax = fig.add_subplot(gs[:, 0])
    # plot estimated and true Ne on this ax
    ax.set_xlabel('Generation')
    ax.set_ylabel('Ne')
    for Ne, lowCI, highCI, label in zip(estNe, lowCIs, highCIs, labelNe):
        ax.plot(np.arange(1, 1+len(Ne)), Ne, label=label)
        ax.fill_between(np.arange(1, 1+len(Ne)), lowCI, highCI, alpha=0.1)
    ax.plot(np.arange(1, 1+len(trueNe)), trueNe, color='black', linestyle= "--")
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize='x-small')

    ### add lines to indicate sampling time
    for _, time in gaps.items():
        ax.axvline(time, 0, 1, color='k', linestyle='--', linewidth=0.5)
            


    ################ plot fit of IBD #######################
    popLabels = nSamples.keys()
    # set up the canvas
    index = 0
    for id1, id2 in itertools.combinations_with_replacement(popLabels, 2):
        id1, id2 = min(id1, id2), max(id1, id2)
        ibd_simulated = []
        for ch in np.arange(1,23):
            ibd_simulated.extend(ibds[(id1, id2)][ch])
        
        # plot the simulated IBD counts
        
        n1, n2 = nSamples[id1], nSamples[id2]
        npairs = n1*n2 if id1 != id2 else n1*(n1-1)/2
        x, _ = np.histogram(ibd_simulated, bins=bins)
        x = np.array(x)/npairs
        
        i, j = index//2, index%2
        ax = fig.add_subplot(gs[i, j+1])
        index += 1
        ax.scatter(midpoint, x, label='simulated', s=7.0, color='grey')

        # plot theoretical expectation        
        meanNumIBD_expectation, _ = singlePop_2tp_given_vecNe(trueNe[max(gaps[id1], gaps[id2]):], L, midpoint, abs(gaps[id1]-gaps[id2]))
        meanNumIBD_expectation = 4*(step/100)*meanNumIBD_expectation
        ax.plot(midpoint, meanNumIBD_expectation, color='black', label='expectation from true Ne', linewidth=0.75)
        # plot fit from estimated Ne
        for Ne, label in zip(estNe, labelNe):
            Ne = np.array(Ne)
            meanNumIBD_expectation, _ = singlePop_2tp_given_vecNe(Ne[max(gaps[id1], gaps[id2]):], L, midpoint, abs(gaps[id1]-gaps[id2]))
            meanNumIBD_expectation = 4*(step/100)*meanNumIBD_expectation
            ax.plot(midpoint, meanNumIBD_expectation, label=label, linewidth=0.75)

        ax.set_yscale('log')
        title = f'({id1}:{gaps[id1]}, {id2}:{gaps[id2]})' if id1 != id2 else f'{id1}:{gaps[id1]}'
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=8)


    ################### save figures #########################
    plt.savefig(f'{outFolder}/{prefix}.fit.png', dpi=300, bbox_inches = "tight")
    plt.clf()


def readNe_withCI(path2file):
    Ne_hist = []
    Ne_low = []
    Ne_high = []
    with open(path2file) as f:
        for line in f:
            _, Ne, low, high = line.strip().split()
            Ne_hist.append(float(Ne))
            Ne_low.append(float(low))
            Ne_high.append(float(high))
    return Ne_hist, Ne_low, Ne_high

def readIBDNe(path2file):
    Ne_hist = []
    Ne_low = []
    Ne_high = []
    with open(path2file) as f:
        f.readline() # skip header line
        line = f.readline()
        while line:
            gen, ne, low, high = line.strip().split()
            Ne_hist.append(float(ne))
            Ne_low.append(float(low))
            Ne_high.append(float(high))
            line = f.readline()
    return Ne_hist, Ne_low, Ne_high