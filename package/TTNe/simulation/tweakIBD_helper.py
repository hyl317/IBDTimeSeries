#### This script contains some functions to "tweak" groundtruth IBD by specified FP rate, power and length bias functions

import numpy as np
import math
import pickle
from analytic_multi import ch_len_dict_default
import pandas as pd

def FP(l, L, Ne):
    """
    This false positive function takes as input, the length of the IBD segment, the length of the chromosome, and the effective population size.
    It gives the false positive rate that is the same as the expected sharing rate for a population of size Ne.
    All length are measured in Morgan.
    """
    return 8*Ne*(1+4*Ne*L)/np.power((1+4*Ne*l),3)

def POWER(x):
    """
    This power function is adapted from Ralph and Coop 2013, eq.1
    x, is the segment length, measured in centimorgan
    """
    return 1 - 1/(1 + 0.025*np.square(x)*np.exp(0.25*x))


# Note: to make simulation and numerical calculation easier, we assume length bias follows from a Gaussian distribution\
# with mean = -0.1246 and sigma^2=1.3 (empirical estimate from benchmark)
from collections import defaultdict
import itertools

def alterIBD_multiTP(path2groundtruth, nSamples, POWER=None, FP=None, Ne=2500, loc=0, scale=math.sqrt(1.3), seed=1):
    # alter the groundtruth IBD according to the given error model
    # nSamples: the dictionary containing sampling cluster index and number of samples in that cluster
    # set random seed for numpy
    np.random.seed(seed)
    binStep = 0.25
    bins = np.arange(4, 20, binStep)
    binMidpoint = (bins[1:] + bins[:-1])/2
    # simualte imperfect power, and, for segments that are detected, add length bias
    ibds_tmp = pickle.load(open(path2groundtruth, 'rb'))
    # preprocess IBD segments
    lst = lambda:defaultdict(list)
    ibds = defaultdict(lst)
    for i in range(len(nSamples)):
        start_i = 0 if i == 0 else 2*sum(nSamples[:i])
        end_i = 2*sum(nSamples[:i+1])
        for j in range(i, len(nSamples)):
            start_j = 0 if j == 0 else 2*sum(nSamples[:j])
            end_j = 2*sum(nSamples[:j+1])
            
            if i != j:
                for hap1, hap2 in itertools.product(np.arange(start_i, end_i), np.arange(start_j, end_j)):
                    hap1, hap2 = min(hap1, hap2), max(hap1, hap2)
                    if (hap1, hap2) in ibds_tmp:
                        for ch, listofibd in ibds_tmp[(hap1, hap2)].items(): 
                            ibds[(i,j)][ch].extend(listofibd)
            else:
                for hap1, hap2 in itertools.combinations(np.arange(start_i, end_i), 2):
                    hap1, hap2 = min(hap1, hap2), max(hap1, hap2)
                    if (hap1, hap2) in ibds_tmp:
                        for ch, listofibd in ibds_tmp[(hap1, hap2)].items(): 
                            ibds[(i,j)][ch].extend(listofibd)

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
                fp_mu = npairs*4*(binStep/100)*FP(binMidpoint/100, ch_len_dict_default[ch]/100, Ne)
                # draw Poisson random var from fp_mu
                nfp = np.random.poisson(fp_mu)
                for i, n in enumerate(nfp):
                    ibds_error[samplePair][ch].extend([binMidpoint[i]]*n)
                    if n != 0:
                        print(f'adding {n} FP to {samplePair} ch{ch} with length {binMidpoint[i]}.')
    # now save the IBD segments with error added
    return ibds, ibds_error
    #pickle.dump(ibds_error, open(f'{path2groundtruth}.{suffix}', 'wb'))


def alterIBD_multiTP_empirical_error_model(df_ibd, sampleCluster2id, POWER=None, FP=None, R=None, seed=1):
    # df_ibd: a pandas dataframe containing IBD segments
    # alter the groundtruth IBD according to the given error model
    # nSamples: the dictionary containing sampling cluster index and number of samples in that cluster
    # set random seed for numpy
    # POWER and FP should be a pickled 1-d interpolator function (from scipy)
    # R should be a pickled KDE object (from scipy)
    np.random.seed(seed)
    binStep = 0.25
    bins = np.arange(4, 20, binStep)
    binMidpoint = (bins[1:] + bins[:-1])/2
    ##### load the error model if provided
    if POWER:
        POWER = pickle.load(open(POWER, 'rb'))
    if FP:
        FP = pickle.load(open(FP, 'rb'))
    if R:
        R = pickle.load(open(R, 'rb'))
    # simualte imperfect power, and, for segments that are detected, add length bias
    

    df_ibd_altered = pd.DataFrame(columns=['iid1', 'iid2', 'ch', 'lengthM'])
    iid1_list = []
    iid2_list = []
    ch_list = []
    lengthCM_list = []

    for index, row in df_ibd.iterrows():
        if POWER is None or np.random.rand() < POWER(100*row['lengthM']):
            # this segment can be detected, now draw its length bias
            if R is None:
                iid1_list.append(row['iid1'])
                iid2_list.append(row['iid2'])
                ch_list.append(row['ch'])
                lengthCM_list.append(100*row['lengthM'])
            else:
                iid1_list.append(row['iid1'])
                iid2_list.append(row['iid2'])
                ch_list.append(row['ch'])
                lengthCM_list.append(100*row['lengthM'] + R.resample(1)[0][0])
    ## add FP segments
    if FP:
        for clst1, clst2 in itertools.combinations_with_replacement(sampleCluster2id.keys(), 2):
            n1, n2 = len(sampleCluster2id[clst1]), len(sampleCluster2id[clst2])
            npairs = n1*n2 if clst1 != clst2 else n1*(n1-1)/2
            for ch in range(1, 23):
                fp_mu = npairs*4*binStep*FP(binMidpoint)*ch_len_dict_default[ch] # be careful with units here
                # draw Poisson random var from fp_mu
                nfp = np.random.poisson(fp_mu)
                for i, n in enumerate(nfp):
                    #print(f'adding {n} FP to ({clst1},{clst2}) ch{ch} with length {binMidpoint[i]}.')
                    for _ in range(n):
                        if clst1 == clst2:
                            # randomly pick two samples from sampleCluster2id[clst1]
                            iid1, iid2 = np.random.choice(sampleCluster2id[clst1], size=2, replace=False)
                        else:
                            iid1, iid2 = np.random.choice(sampleCluster2id[clst1]), np.random.choice(sampleCluster2id[clst2])
                        iid1_list.append(iid1)
                        iid2_list.append(iid2)
                        ch_list.append(ch)
                        lengthCM_list.append(binMidpoint[i])

    # now save the IBD segments with error added
    df_ibd_altered['iid1'] = iid1_list
    df_ibd_altered['iid2'] = iid2_list
    df_ibd_altered['ch'] = ch_list
    df_ibd_altered['lengthM'] = lengthCM_list
    df_ibd_altered['lengthM'] = df_ibd_altered['lengthM']/100
    return df_ibd_altered







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