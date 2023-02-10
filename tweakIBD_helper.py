#### This script contains some functions to "tweak" groundtruth IBD by specified FP rate, power and length bias functions

import numpy as np
import math
import pickle

def FP(x):
    """
    This false positive function is taken from Ralph and Coop, 2013. 
    This function should be interpreted as the mean density of false positives xcM long per haplotype pair and per cM.
    See eq.2 on page 13 of the manuscript's pdf
    """
    return np.exp(-13 - 2*x + 4.3*np.sqrt(x))/4

def POWER(x):
    """
    This power function is taken from Ralph and Coop 2013, eq.1
    """
    return 1 - 1/(1 + 0.077*np.square(x)*np.exp(0.54*x))


def power(x):
    """
    This power function is defined so that it has 50% power at 4cM and 99.9% power at 20cM
    Only defined for x >=6 and x<=20
    """
    return 0.5 + 0.499/np.sqrt(16)*np.sqrt(x-6)

def power2(x):
    """
    A function for power curve that is defined for all positive x.
    """
    return 1 - np.exp(-0.2*x)

# Note: to make simulation and numerical calculation easier, we assume length bias follows from a Gaussian distribution\
# with mean = -0.1246 and sigma^2=1.3 (empirical estimate from benchmark)

def alterIBD_multiTP(path2groundtruth, nSamples, ch_len_dict, POWER=None, FP=None, FPscale=10, loc=0, scale=math.sqrt(1.3), suffix='error'):
    # alter the groundtruth IBD according to the given error model
    # nSamples: the dictionary containing sampling cluster index and number of samples in that cluster
    binStep = 0.005
    bins = np.arange(4, 20, binStep)
    binMidpoint = (bins[1:] + bins[:-1])/2
    # simualte imperfect power, and, for segments that are detected, add length bias
    ibds = pickle.load(open(path2groundtruth, 'rb'))
    ibds_error = {}
    for samplePair in ibds.keys():
        ibds_error[samplePair] = {}
        if samplePair[0] == samplePair[1]:
            n = nSamples[samplePair[0]]
            npairs = n*(n-1)/2
        else:
            n1, n2 = nSamples[samplePair[0]], nSamples[samplePair[1]]
            npairs = n1*n2
        for ch, ibdlist in ibds[samplePair].items():
            ibds_error[samplePair][ch] = []
            for ibd in ibdlist:
                if POWER is None or np.random.rand() < POWER(ibd):
                    # this segment can be detected, now draw its length bias
                    if loc == 0 and scale == 0:
                        ibds_error[samplePair][ch].append(ibd)
                    else:
                        ibds_error[samplePair][ch].append(ibd + np.random.normal(loc=loc, scale=scale))
            ## add FP segments
            if not FP is None:
                fp_mu = npairs*4*ch_len_dict[ch]*0.005*FP(binMidpoint)*FPscale
                # draw Poisson random var from fp_mu
                nfp = np.random.poisson(fp_mu)
                for i, n in enumerate(nfp):
                    ibds_error[samplePair][ch].extend([binMidpoint[i]]*n)
                    if n != 0:
                        print(f'adding {n} FP to {samplePair} ch{ch} with length {binMidpoint[i]}.')
    # now save the IBD segments with error added
    pickle.dump(ibds_error, open(f'{path2groundtruth}.{suffix}', 'wb'))


def alterIBD_singlePair(path2groundtruth, nPairs, chromLength, POWER=None, FP=None, FPscale=10, loc=0, scale=math.sqrt(1.3), suffix='error'):
    # alter the groundtruth IBD according to the given error model
    # nPairs: number of haplotype pairs
    # chromLength: length of haplotype pair, measured in cM
    binStep = 0.005
    bins = np.arange(4, 20, binStep)
    binMidpoint = (bins[1:] + bins[:-1])/2
    # simualte imperfect power, and, for segments that are detected, add length bias
    ibds = pickle.load(open(path2groundtruth, 'rb'))
    ibds_error = []
    for ibd in ibds:
        if POWER is None or np.random.rand() < POWER(ibd):
            # this segment can be detected, now draw its length bias
            if loc == 0 and scale == 0:
                ibds_error.append(ibd)
            else:
                ibds_error.append(np.maximum(0.0, ibd + np.random.normal(loc=loc, scale=scale)))
            ## add FP segments
    fp_mu = nPairs*chromLength*0.005*FP(binMidpoint)*FPscale
    # draw Poisson random var from fp_mu
    nfp = np.random.poisson(fp_mu)
    for i, n in enumerate(nfp):
        ibds_error.extend([binMidpoint[i]]*n)
        if n != 0:
            print(f'adding {n} FP to with length {binMidpoint[i]}.')
    # now save the IBD segments with error added
    pickle.dump(ibds_error, open(f'{path2groundtruth}.{suffix}', 'wb'))


###### alter IBD for twoPopIM

def alterIBD_twoPopIM(path2groundtruth, nSamples, ch_len_dict, POWER=None, FP=None, FPscale=10, loc=0, scale=math.sqrt(1.3), suffix='error'):
    # alter the groundtruth IBD according to the given error model
    # nSamples: the dictionary containing sampling cluster index and number of samples in that cluster
    binStep = 0.005
    bins = np.arange(4, 20, binStep)
    binMidpoint = (bins[1:] + bins[:-1])/2
    # simualte imperfect power, and, for segments that are detected, add length bias
    ibds = pickle.load(open(path2groundtruth, 'rb'))
    ibds = ibds[(0,0)]
    ibds_error = {}
    npairs = nSamples*nSamples

    for ch, ibdlist in ibds.items():
        ibds_error[ch] = []
        for ibd in ibdlist:
            if POWER is None or np.random.rand() < POWER(ibd):
                # this segment can be detected, now draw its length bias
                if loc == 0 and scale == 0:
                    ibds_error[ch].append(ibd)
                else:
                    ibds_error[ch].append(ibd + np.random.normal(loc=loc, scale=scale))
        ## add FP segments
        if not FP is None:
            fp_mu = npairs*4*ch_len_dict[ch]*0.005*FP(binMidpoint)*FPscale
            # draw Poisson random var from fp_mu
            nfp = np.random.poisson(fp_mu)
            for i, n in enumerate(nfp):
                ibds_error[ch].extend([binMidpoint[i]]*n)
                if n != 0:
                    print(f'adding {n} FP to ch{ch} with length {binMidpoint[i]}.')
    # now save the IBD segments with error added
    pickle.dump(ibds_error, open(f'{path2groundtruth}.{suffix}', 'wb'))