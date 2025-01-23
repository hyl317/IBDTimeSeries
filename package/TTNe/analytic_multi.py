import sys
import numpy as np
import itertools
from collections import defaultdict
from scipy.optimize import minimize
from scipy.ndimage import shift
from scipy.stats import norm
from TTNe.analytic import singlePop_2tp_given_Ne_negLoglik, \
    singlePop_2tp_given_vecNe_negLoglik_noPenalty,\
    singlePop_2tp_given_vecNe_DevStat_noPenalty
from TTNe.inference_utility import get_ibdHistogram_by_sampleCluster
from TTNe.plot import plot_pairwise_fitting, plot_pairwise_TMRCA
import matplotlib.pyplot as plt
import pickle
import warnings
from pathlib import Path
import pandas as pd
import itertools
import os
import time
import math
from numba import prange
from tqdm import tqdm
import multiprocessing as mp

ch_len_dict_default = {1:286.279, 2:268.840, 3:223.361, 4:214.688, 5:204.089, 6:192.040, 7:187.221, 8:168.003, 9:166.359, \
        10:181.144, 11:158.219, 12:174.679, 13:125.706, 14:120.203, 15:141.860, 16:134.038, 17:128.491, 18:117.709, \
        19:107.734, 20:108.267, 21:62.786, 22:74.110}


def singlePop_MultiTP_given_Ne_negLoglik(Ne, histograms, binMidpoint, chrlens, dates, nHaplotypePairs, timeBoundDict, s=0, e=-1, FP=None, R=None, POWER=None):
    accu = 0
    for (clusterID1, clusterID2), nHaplotypePair in nHaplotypePairs.items():
        if nHaplotypePair == 0:
            continue
        clusterID1, clusterID2 = min(clusterID1, clusterID2), max(clusterID1, clusterID2)
        if isinstance(FP, dict):
            FP_ = FP[(clusterID1, clusterID2)]
        else:
            FP_ = FP
        accu += singlePop_2tp_given_Ne_negLoglik(Ne, histograms[(clusterID1, clusterID2)], 
                binMidpoint, chrlens, dates[clusterID1], dates[clusterID2], nHaplotypePair,
                [(timeBoundDict[clusterID1]), (timeBoundDict[clusterID2])], s, e, FP_, R, POWER)
    return accu

def inferConstNe_singlePop_MultiTP(histograms, binMidpoint, dates, nHaplotypePairs, Ninit, chrlens, timeBoundDict, s=0, e=-1, FP=None, R=None, POWER=None):
    # given the observed ibd segments, estimate the Ne assuming a constant effective Ne
    # dates: dictionary, where the key is the sampling cluter index and the value is the sampling time (in generations ago)
    # nSamples: dictionary, where the key is the sampling cluster index and the value is the number of diploid samples within each cluster

    kargs = (histograms, binMidpoint, chrlens, dates, nHaplotypePairs, timeBoundDict, s, e, FP, R, POWER)
    res = minimize(singlePop_MultiTP_given_Ne_negLoglik, Ninit, args=kargs, method='L-BFGS-B', bounds=[(10, 5e6)])
    return res.x[0]

def singlePop_MultiTP_given_vecNe_DevStat(Ne, histograms, binMidpoint, chrlens, dates, nHaplotypePairs, Tmax, timeBoundDict, \
                                            s=0, e=-1, FP=None, R=None, POWER=None):
    accu = 0.0
    for (clusterID1, clusterID2), nHaplotypePair in nHaplotypePairs.items():
        if nHaplotypePair == 0:
            continue
        clusterID1, clusterID2 = min(clusterID1, clusterID2), max(clusterID1, clusterID2)
        accu_, _ = singlePop_2tp_given_vecNe_DevStat_noPenalty(Ne, histograms[(clusterID1, clusterID2)],
            binMidpoint, chrlens, dates[clusterID1], dates[clusterID2], Tmax, nHaplotypePair, 
            [timeBoundDict[clusterID1], timeBoundDict[clusterID2]], s=s, e=e, FP=FP, R=R, POWER=POWER)
        accu += accu_
    return accu

def singlePop_MultiTP_given_vecNe_negLoglik(Ne, histograms, binMidpoint, chrlens, dates, nHaplotypePairs, Tmax, alpha, beta, timeBoundDict, \
                                            s=0, e=-1, FP=None, R=None, POWER=None, tail=False):
    # return the negative loglikelihood of Ne, plus the penalty term
    # G: a vector of chromosome length
    # gaps: dictionary, where the key is the sampling cluter index and the value is the sampling time
    # nSamples: dictionary, where the key is the sampling cluster index and the value is the number of diploid samples within each cluster
    accu = 0.0
    grad = np.zeros_like(Ne)

    ################################ unify computation for both within and between sample cluster ############################
    for (clusterID1, clusterID2), nHaplotypePair in nHaplotypePairs.items():
        if nHaplotypePair == 0:
            continue
        clusterID1, clusterID2 = min(clusterID1, clusterID2), max(clusterID1, clusterID2)
        if isinstance(FP, dict):
            FP_ = FP[(clusterID1, clusterID2)]
        else:
            FP_ = FP
        accu_, grad_ = singlePop_2tp_given_vecNe_negLoglik_noPenalty(Ne, histograms[(clusterID1, clusterID2)],
            binMidpoint, chrlens, dates[clusterID1], dates[clusterID2], Tmax, nHaplotypePair, 
            [timeBoundDict[clusterID1], timeBoundDict[clusterID2]], s=s, e=e, FP=FP_, R=R, POWER=POWER, tail=tail)
        accu += accu_
        grad += grad_
    
    if alpha != 0:
        penalty1 = alpha*np.sum(np.diff(np.log(Ne), n=2)**2)
        accu += penalty1

        # add gradient due to the penalty term
        Ne_ = np.log(Ne)
        N_left2 = shift(Ne_, -2, cval=0)
        N_left1 = shift(Ne_, -1, cval=0)
        N_right2 = shift(Ne_, 2, cval=0)
        N_right1 = shift(Ne_, 1, cval=0)
        penalty_grad1 = 12*Ne_ - 8*(N_left1 + N_right1) + 2*(N_left2 + N_right2)
        penalty_grad1[0] = 2*Ne_[0]-4*Ne_[1]+2*Ne_[2]
        penalty_grad1[1] = 10*Ne_[1]-4*Ne_[0]-8*Ne_[2]+2*Ne_[3]
        penalty_grad1[-1] = 2*Ne_[-1]-4*Ne_[-2]+2*Ne_[-3]
        penalty_grad1[-2] = 10*Ne_[-2]-4*Ne_[-1]-8*Ne_[-3]+2*Ne_[-4]
        grad += alpha*penalty_grad1/Ne

    if beta != 0:
        ################################### weighted first derivative penalty 
        sigma = (len(Ne)-1)/3
        weights = norm.pdf(np.arange(1, len(Ne)), loc=len(Ne)-1, scale=sigma)
        weights = weights/weights[-1]

        penalty2 = beta*np.sum((np.diff(np.log(Ne), n=1)**2)*weights)
        accu += penalty2

        Ne_ = np.log(Ne)
        penalty_grad2 = np.zeros_like(Ne_)
        penalty_grad2[1:-1] = 2*weights[1:]*Ne_[1:-1] + 2*weights[:-1]*Ne_[1:-1] - 2*weights[:-1]*shift(Ne_, 1)[1:-1] - 2*weights[1:]*shift(Ne_, -1)[1:-1]
        penalty_grad2[0] = -2*weights[0]*(Ne_[1] - Ne_[0])
        penalty_grad2[-1] = 2*weights[-1]*(Ne_[-1] - Ne_[-2])

        # penalty_grad2 = 4*Ne_ - 2*shift(Ne_, 1, cval=0) - 2*shift(Ne_, -1, cval=0)
        # penalty_grad2[0] = -(2*Ne_[1] - 2*Ne_[0])
        # penalty_grad2[-1] = 2*Ne_[-1] - 2*Ne_[-2]
        grad += beta*penalty_grad2/Ne
        ###########################################################################
        ### diff to estimated constant Ne #########################################
        # penalty3 = beta*np.sum((np.log(Ne) - np.log(Nconst))**2)
        # accu += penalty3
        # grad += beta*2*(np.log(Ne) - np.log(Nconst))/Ne
        

    return accu, grad

def bootstrap_single_run(histograms, chrlens, dates, nHaplotypePairs, 
        timeBoundDict, Ninit=500, Tmax=100,
        minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0,
        step=0.1, alpha=1e-6, beta=1e-4, method='l2', FP=None, R=None, POWER=None, verbose=False):
    # perform one bootstrap resampling
    bins = np.arange(minL_calc, maxL_calc+step, step=step)
    binMidpoint = (bins[1:] + bins[:-1])/2
    ###################### First, infer a const Ne ################################
    bins_infer = np.arange(minL_infer, maxL_infer+step, step)
    binMidpoint_infer = (bins_infer[1:] + bins_infer[:-1])/2
    s = np.where(np.isclose(binMidpoint, binMidpoint_infer[0]))[0][0]
    e = np.where(np.isclose(binMidpoint, binMidpoint_infer[-1]))[0][0]
    if (not FP is None) and (not R is None) and (not POWER is None):
        if isinstance(FP, dict):
            for _, v in FP.items():
                assert(len(v) == len(binMidpoint))
        else:
            assert(len(FP) == len(binMidpoint))
        assert(len(POWER) == len(binMidpoint))
        assert(R.shape[0] == R.shape[1] == len(binMidpoint))
    Nconst = inferConstNe_singlePop_MultiTP(histograms, binMidpoint, dates, nHaplotypePairs, Ninit, chrlens, timeBoundDict, s, e, FP, R, POWER)
    if verbose:
        print(f'estimated constant Ne: {Nconst}')
    ################################################################################
    t_min = np.min(list(dates.values()))
    t_max = np.max(list(dates.values()))
    i_min = -1
    i_max = -1
    for k, v in dates.items():
        if v == t_min:
            i_min = k
        if v == t_max:
            i_max = k
    assert(i_min != -1)
    assert(i_max != -1)
    t_min += timeBoundDict[i_min][0] # here t_min is either 0 or negative
    t_max += timeBoundDict[i_max][1]
    ### shift each sample cluster's time by |t_min|
    dates_adjusted = {}
    for k, v in dates.items():
        dates_adjusted[k] = v + abs(t_min)

    
    NinitVec = np.random.normal(Nconst, Nconst/25, Tmax + (t_max - t_min))
    ###################### the following is just some diagnostics for myself ############################
    for clusterID in dates.keys():
        histograms_, nHaplotypePairs_ = {}, {}
        histograms_[(clusterID, clusterID)] = histograms[(clusterID, clusterID)]
        nHaplotypePairs_[(clusterID, clusterID)] = nHaplotypePairs[(clusterID, clusterID)]
        constNe = inferConstNe_singlePop_MultiTP(histograms_, binMidpoint, {clusterID:0}, nHaplotypePairs_, Ninit, chrlens, timeBoundDict, s, e, FP, R, POWER)
        if verbose:
            print(f'constNe for {clusterID} is {constNe}')
    #####################################################################################################
            
    kargs = (histograms, binMidpoint, chrlens, dates_adjusted, nHaplotypePairs, Tmax, alpha, beta, timeBoundDict, s, e+1, FP, R, POWER)  
    if method == 'l2':
        t1 = time.time()
        res = minimize(singlePop_MultiTP_given_vecNe_negLoglik, NinitVec, \
                args=kargs, method='L-BFGS-B', jac=True, bounds=[(10, 5e6) for i in range(Tmax + (t_max - t_min))], \
                options={'maxfun': 50000, 'maxiter':50000, 'gtol':1e-8, 'ftol':1e-8})
        if verbose:
            print(res)
            print(f'elapsed time for optimization: {time.time() - t1:.3f}s')
        return res.x
    else:
        raise RuntimeError('Unsupported Method')

def bootstrap_single_run_wrapper(args):
    return bootstrap_single_run(*args)

def bootstrap(func, histogram_list, chrlens_list, nprocess, *args):
    """
    fullResults: whether to return the full results of individual bootstrap resampling
    """
    # perform bootstrap resampling, and then return the 95% CI of the estimated Ne trajectory
    time1 = time.time()
    args = list(args)
    params = [[histogram, chrlens, *args] for histogram, chrlens in zip(histogram_list, chrlens_list)]
    with mp.Pool(processes = nprocess) as pool:
        results = list(tqdm(pool.imap(func, params), total=len(chrlens_list)))
    results_aggregate = np.zeros((len(results), len(results[0])))
    for i in range(len(results)):
        results_aggregate[i] = results[i]
    results_aggregate_sorted = np.sort(results_aggregate, axis=0)
    print(f'bootstrap takes {time.time() - time1:.3f}s')

    return results_aggregate_sorted

    

def prepare_input(ibds_by_chr, ch_ids, ch_len_dict, clusterIDs, minL=4.0, maxL=20.0, step=0.1):
    ibds = defaultdict(list)
    for pair, ibd_dict_pair in ibds_by_chr.items():
        for ch in ch_ids:
            if ibd_dict_pair.get(ch):
                ibds[pair].extend(ibd_dict_pair[ch])

    bins = np.arange(minL, maxL+step, step=step)
    binMidpoint = (bins[1:] + bins[:-1])/2
    histograms = {}
    for id1, id2 in itertools.combinations_with_replacement(clusterIDs, 2):
        histogram, _ = np.histogram(ibds[(min(id1, id2), max(id1, id2))], bins=bins)
        histograms[(min(id1, id2), max(id1, id2))] = histogram
    
    chrlens = [ch_len_dict[ch] for ch in ch_ids]
    return histograms, binMidpoint, np.array(chrlens)


def inferVecNe_singlePop_MultiTP(df_ibd, sampleCluster2id, dates, ch_len_dict=None, timeBoundDict=None, Ninit=500, Tmax=100, \
        minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0,
        step=0.1, alpha=2500, beta=250, method='l2', FP=None, R=None, POWER=None, plot=False, prefix="", \
        verbose=True, doBootstrap=True, autoHyperParam=False, outFolder='.', parallel=True, nprocess=10):
    """
    Parameter
    ---------
    df_ibd: pandas dataframe
        A dataframe with each row being a IBD segment. It must have at least 4 columns: "iid1", "iid2", "ch", "lengthM". Note that the length is in Morgan.
        It is fine to have more than these columns, but those extra columns will not be used.
    sampleCluster2id: dictionary
        A dictionary mapping sample clusterID to sample IID that belong to that cluster. Samples in the same sample cluster should be dated to the same or very similar date.
    dates: dictionary
        A dictionary mapping sample clusterID to the date (in generations ago) of that sample cluster. The most recent sample cluster must be labeled as generation 0. All other dates are relative to this most recent date.
    nSamples: dictionary
        A dictionary mapping sample clusterID to the number of samples in that cluster.
    timeBound: dict
        A dictionary specifying the dating uncertainty of each of the sampling cluster. For example, 1:(-2,3) means that the dating of the sample cluster 1
        is distributed from 2 generations more recent, to 3 generations older than the time point given in the parameter gaps. We assume a uniform distribution across this range.
    
    """
    if ((FP is None) or (R is None) or (POWER is None)) and (minL_calc != minL_infer or maxL_calc != maxL_infer):
        warnings.warn('Error model not provided... Setting the length range used for calculation and inference to be the same.')
        minL_calc = minL_infer
        maxL_calc = maxL_infer
    print(f'min IBD length: {np.min(df_ibd["lengthM"].values)}')
    # estimate Ne using the original data
    if ch_len_dict is None:
        ch_len_dict = ch_len_dict_default
    ch_ids = [k for k, v in ch_len_dict.items()]
    chrlens = np.array([l for _, l in ch_len_dict.items()])
    if not timeBoundDict:
        timeBoundDict = {}
        for clusterID, _ in dates.items():
            timeBoundDict[clusterID] = (0, 0)
    assert(np.min(list(dates.values())) == 0) # the most recent sample cluster must be labeled as generation 0
    # calculate # of haplotype pairs for each pairs of sample clusters
    nHaplotypePairs = {}
    for clusterID1, clusterID2 in itertools.combinations_with_replacement(dates.keys(), 2):
        clusterID1, clusterID2 = min(clusterID1, clusterID2), max(clusterID1, clusterID2)
        nSample1 = len(sampleCluster2id[clusterID1])
        nSample2 = len(sampleCluster2id[clusterID2])
        if clusterID1 == clusterID2:
            nHaplotypePairs[(clusterID1, clusterID2)] = 2*nSample1*(2*nSample1-2)/2
        else:
            nHaplotypePairs[(clusterID1, clusterID2)] = 2*nSample1*2*nSample2
    print(f'number of haplotype pairs: {nHaplotypePairs}')
    #### do hyperparameter search if needed
    if autoHyperParam:
        # t1 = time.time()
        # _, _, alpha = hyperparam_kfold_parallel(df_ibd, sampleCluster2id, dates, chrlens, timeBoundDict,
        #             kfold=5, Ninit=Ninit, Tmax=Tmax,
        #             minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer,
        #             FP=FP, R=R, POWER=POWER, step=step, beta=beta, 
        #             prefix=prefix, outfolder=outFolder, save=True)
        # print(f'parallel hyperparameter search takes {time.time() - t1:.3f}s')
        t1 = time.time()
        alpha = hyperparam_kfold(df_ibd, sampleCluster2id, dates, chrlens, timeBoundDict, 
                    kfold=5, Ninit=Ninit, Tmax=Tmax,
                    minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer,
                    FP=FP, R=R, POWER=POWER, step=step, beta=beta, 
                    prefix=prefix, outfolder=outFolder, save=True, parallel=parallel, nprocess=nprocess, method=method)
        print(f'hyperparameter search takes {time.time() - t1:.3f}s')

    bins = np.arange(minL_calc, maxL_calc+step, step=step)
    histograms = get_ibdHistogram_by_sampleCluster(df_ibd, sampleCluster2id, bins, ch_ids)
    Ne = bootstrap_single_run(histograms, chrlens, dates, nHaplotypePairs, timeBoundDict, Ninit=Ninit, Tmax=Tmax, \
        minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, \
        step=step, alpha=alpha, beta=beta, method=method, FP=FP, R=R, POWER=POWER, verbose=verbose)
    if plot:
        if len(prefix) > 0:
            plot_prefix = "pairwise." + prefix
        else:
            plot_prefix = "pairwise"
        min_plot = max(6.0, math.floor(np.min(100*df_ibd['lengthM'].values)))
        print(f'minL for plotting: {min_plot}')
        plot_pairwise_fitting(df_ibd, sampleCluster2id, dates, nHaplotypePairs, ch_len_dict, Ne, outFolder, timeBoundDict, prefix=plot_prefix, \
            minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL=maxL_infer, step=step, minL_plot=min_plot, FP=FP, R=R, POWER=POWER)
        plot_pairwise_TMRCA(dates, Ne, Tmax, outFolder, prefix=plot_prefix, minL=minL_infer, maxL=maxL_infer, step=0.25)

    if doBootstrap:
        # start bootstrapping
        nresample = 200
        ch_ids = [k for k, v in ch_len_dict.items()]
        resample_chrs = [ np.random.choice(ch_ids, size=len(ch_ids)) for i in range(nresample)]
        if not parallel:
            bootstrap_results = np.full((nresample, len(Ne)), np.nan)
            time1 = time.time()
            for i in tqdm(range(nresample)):
                histograms = get_ibdHistogram_by_sampleCluster(df_ibd, sampleCluster2id, bins, resample_chrs[i])
                chrlens = np.array([ch_len_dict[ch] for ch in resample_chrs[i]])
                bootstrap_results[i] = bootstrap_single_run(histograms, chrlens, 
                    dates, nHaplotypePairs, timeBoundDict, Ninit=Ninit, Tmax=Tmax, 
                    minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer,
                    step=step, alpha=alpha, beta=beta, method=method, FP=FP, R=R, POWER=POWER)
            bootstrap_results_sorted = np.sort(bootstrap_results, axis=0)
            print(f'sequential bootstrap takes {time.time() - time1:.3f}s')
        else:
            time1 = time.time()
            histogram_list = [ get_ibdHistogram_by_sampleCluster(df_ibd, sampleCluster2id, bins, resample_chr) for resample_chr in resample_chrs]
            ch_len_list = [np.array([ch_len_dict[ch] for ch in resample_chr]) for resample_chr in resample_chrs]
            bootstrap_results_sorted = bootstrap(bootstrap_single_run_wrapper, histogram_list, ch_len_list, nprocess, dates, nHaplotypePairs, timeBoundDict, Ninit, Tmax, \
                minL_calc, maxL_calc, minL_infer, maxL_infer, step, alpha, beta, method, FP, R, POWER)
            print(f'parallel bootstrap takes {time.time() - time1:.3f}s')

        # create an empty pandas dataframe with three columns named Ne, lowCI, upCI
        df = pd.DataFrame(columns=['Generations', 'Ne', 'lowCI', 'upCI'])
        df['Generations'] = np.arange(1, len(Ne)+1)
        df['Ne'] = Ne
        df['lowCI'] = bootstrap_results_sorted[int(0.025*nresample)]
        df['upCI'] = bootstrap_results_sorted[int(0.975*nresample)]
    else:
        df = pd.DataFrame(columns=['Generations', 'Ne'])
        df['Generations'] = np.arange(1, len(Ne)+1)
        df['Ne'] = Ne
    return df


def inferVecNe_singlePop_MultiTP_withMask(path2IBD, path2SampleAge, path2ChrDelimiter, path2mask=None, \
        Ninit=500, Tmax=100, minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, step=0.25, alpha=2500, beta=250, method='l2', \
        FP=None, R=None, POWER=None, generation_time=29, minSample=10, maxSample=np.inf, merge_level=5, \
        prefix="", doBootstrap=True, autoHyperParam=True, outFolder='.', dryrun=False, plot=True, parallel=True, nprocess=10):
    ###### read in a list of samples with their mean BP. 
    sampleAgeDict = {}
    youngest_age = np.inf
    with open(path2SampleAge) as f:
        for line in f:
            iid, BPage, *_ = line.strip().split()
            BPage = int(BPage)
            sampleAgeDict[iid] = BPage
            if BPage < youngest_age:
                youngest_age = BPage
    sampleAgeBinDict = {} # each bin differ by 5 generations
    for iid, BPage in sampleAgeDict.items():
        sampleAgeBinDict[iid] = (int(BPage - youngest_age)/generation_time)//merge_level # we bin sample age by 5 generations
    maxBinIndex = max([int(v) for _, v in sampleAgeBinDict.items()])
    dates = {i: merge_level*i for i in range(0, maxBinIndex+1)}
    nSamples = {i: sum(v == i for _, v in sampleAgeBinDict.items()) for i in range(0, maxBinIndex+1)}

    exSampleCluster = set()
    sampleCluster2id = {}
    for i in range(0, maxBinIndex+1):
        n = nSamples[i]
        if n < minSample or n > maxSample:
            exSampleCluster.add(i)
        else:
            sampleCluster2id[i] = [k for k, v in sampleAgeBinDict.items() if v == i]
        print(f'{nSamples[i]} samples in age bin {i}({youngest_age + i*merge_level*generation_time} - {youngest_age + (i+1)*merge_level*generation_time}BP): \n{[k for k, v in sampleAgeBinDict.items() if v == i]}')
    
    for i in exSampleCluster:
        del nSamples[i]
        del dates[i]
    print(f'Bins {[i for i, _ in dates.items()]} will be used for inference.')
    print(sampleCluster2id)
    
    timeOffset = np.min(list(dates.values()))
    for i in dates.keys():
        dates[i] -= timeOffset
        
    ################################ End of allocating samples to different time periods #######################
    
    ################################ masking out IBD from potentially problematic genomic regions ################
    chrDelimiter = {}
    with open(path2ChrDelimiter) as f:
        for line in f:
            chr, start, end = line.strip().split()
            start, end = float(start), float(end)
            start = max(0.0, start)
            chrDelimiter[int(chr)] = (float(start), float(end))

    if path2mask != None:
        masks = defaultdict(lambda: [])
        with open(path2mask) as f:
            for line in f:
                chr, bp_start, bp_end, cm_start, cm_end = line.strip().split()
                masks[int(chr)].append((float(cm_start), float(cm_end)))
    
        #### now generate a set of "artificial" chromosomes based on the original chromosome and the provided masks
        counter = 0
        # this dictionary records the mapping from the artificial chrom to the original chrom
        # in the form of value mapped to a key taking the form: (original chrom index, start_cm, end_cm)
        chr_span_dict = {}  
        map = {}

        for ch, delim in chrDelimiter.items():
            ch_start, ch_end = delim
            map[ch] = counter
            if len(masks[ch]) == 0:
                chr_span_dict[counter] = (ch, ch_start, ch_end)
                counter += 1
            else:
                mask_start, mask_end = -np.inf, np.inf
                for mask in masks[ch]:
                    mask_start,  mask_end = mask
                    assert(mask_start >= ch_start and mask_end <= ch_end)
                    if mask_start - ch_start >= 40:
                        chr_span_dict[counter] = (ch, ch_start, mask_start)
                        counter += 1
                    ch_start = mask_end
            
                if ch_end - mask_end >= 40:
                    chr_span_dict[counter] = (ch, mask_end, ch_end)
                    counter += 1
        map[ch+1] = counter
        print(f'After masking, {len(chr_span_dict)} artificial chromosomes are created from {len(chrDelimiter)} original chromosomes.')
        ch_len_dict = {k: v[2]-v[1] for k, v in chr_span_dict.items()}
        spanlist = [(v[1], v[2]) for k, v in chr_span_dict.items()]
    
        #### Now mask IBD segments and assign them to the correct artificial chromosomes
        iid1_list, iid2_list, ch_list, genetic_length_list = [], [], [], []
        df_ibd = pd.read_csv(path2IBD, sep='\t')
        for _, row in df_ibd.iterrows():
            iid1, iid2, ch, ibd_start, ibd_end, length = row['iid1'], row['iid2'], row['ch'], 100*row['StartM'], 100*row['EndM'], 100*row['lengthM']
            if sampleAgeBinDict.get(iid1) == None or sampleAgeBinDict.get(iid2) == None:
                continue
            for i, span in enumerate(spanlist[map[ch]:map[ch+1]]):
                # ask whether the current segment being read has any overlap with this span of interest
                span_start, span_end = span
                if ibd_end <= span_start or ibd_start >= span_end:
                    continue
                else:
                    ibd_start_masked, ibd_end_masked = max(span_start, ibd_start), min(span_end, ibd_end)
                    iid1_list.append(iid1)
                    iid2_list.append(iid2)
                    ch_list.append(i+map[ch])
                    genetic_length_list.append((ibd_end_masked - ibd_start_masked)/100)
        df_ibd = pd.DataFrame({'iid1':iid1_list, 'iid2':iid2_list, 'ch':ch_list, 'lengthM':genetic_length_list})
    else:
        ########## no mask file provided, thus no masking will be performed ###################
        print(f'no mask file provided, all segments will be used for inference')
        ch_len_dict = {k:v[1]-v[0] for k, v in chrDelimiter.items()}
        df_ibd = pd.read_csv(path2IBD, sep='\t')
        #print(f'######## {np.min(df_ibd["lengthM"].values)}')

    ### end of preprocessing of input data, ready to start the inference
    if not dryrun:
        return inferVecNe_singlePop_MultiTP(df_ibd, sampleCluster2id, dates, ch_len_dict, Ninit=Ninit, Tmax=Tmax, \
                minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, alpha=alpha, beta=beta, \
                method=method, FP=FP, R=R, POWER=POWER, plot=plot, 
                prefix=prefix, autoHyperParam=autoHyperParam, 
                doBootstrap=doBootstrap, outFolder=outFolder,
                parallel=parallel, nprocess=nprocess)
    else:
        return df_ibd, sampleCluster2id, dates, ch_len_dict, FP, R, POWER


def hyperparam_opt2(ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict, Ninit=500, Tmax=100, \
                    minL_calc=2.0, maxL_calc=22, minL_infer=6.0, maxL_infer=20.0,
                    FP=None, R=None, POWER=None, step=0.1, beta=250, prefix="", outfolder="", history=False):
    alphas = np.logspace(np.log10(50), 7, 30)
    loglik = np.zeros(len(alphas))
    Nes = []
    chs = np.arange(1, 23)
    for i, alpha in enumerate(alphas):
        Ne = bootstrap_single_run(chs, ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict, Ninit, Tmax, \
                                  minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, alpha=alpha, beta=beta,\
                                    FP=FP, R=R, POWER=POWER)
        histograms, binMidpoint, G = prepare_input(ibds_by_chr, chs, ch_len_dict, nSamples.keys(), \
                                        minL=minL_calc, maxL=maxL_calc, step=step)
        bins_infer = np.arange(minL_infer, maxL_infer+step, step)
        binMidpoint_infer = (bins_infer[1:] + bins_infer[:-1])/2
        s = np.where(np.isclose(binMidpoint, binMidpoint_infer[0]))[0][0]
        e = np.where(np.isclose(binMidpoint, binMidpoint_infer[-1]))[0][0]
        loglike, _ = singlePop_MultiTP_given_vecNe_negLoglik(Ne, histograms, binMidpoint, G, gaps, nSamples, Tmax, 0, 0, timeBoundDict,\
                                                                    s=s, e=e, FP=FP, R=R, POWER=POWER)
        loglik[i] = -loglike
        Nes.append(Ne)
    plt.plot(alphas, loglik, marker='o')
    plt.xlabel('alpha')
    plt.ylabel('loglikelihood')
    plt.xscale('log')

    delimiter = '.'
    plt.savefig(os.path.join(outfolder, delimiter.join([s for s in [prefix, "hyperparam2.pdf"] if s.strip()])), dpi=300)
    # save the numpy array loglik to a file
    np.save(os.path.join(outfolder, delimiter.join([s for s in [prefix, 'hyperparam2.npy'] if s.strip()])), loglik)
    if history:
       pickle.dump(Nes, open(os.path.join(outfolder, \
                       delimiter.join([s for s in [prefix, "hyperparam2.Nes.pickle"] if s.strip()])), 'wb'))
    
    ########### now, determine the best hyperparam ################
    # first, find the largest alpha that gives a loglikelihood within 1% of the maximum
    max_loglik = np.max(loglik)
    max_alpha_index = np.argmax(loglik)
    print(f'max_loglik={max_loglik}, max_alpha_index={max_alpha_index}, alpha_with_ML={alphas[max_alpha_index]}')
    # find the largest alpha that within 1.1 from max_loglik
    opt_alpha_index = np.max(np.where(loglik > max_loglik - 1.2)[0])
    opt_alpha = alphas[opt_alpha_index]
    print(f'route 1: opt_alpha={opt_alpha}')

    return max(2500, opt_alpha)


def hyperparam_opt_DevStat(ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict, Ninit=500, Tmax=100, \
                    minL_calc=2.0, maxL_calc=22, minL_infer=6.0, maxL_infer=20.0,
                    FP=None, R=None, POWER=None, step=0.1, beta=250, prefix="", outfolder="", history=False):
    alphas = np.logspace(np.log10(50), 7, 30)
    devstats = np.zeros(len(alphas))
    Nes = []
    chs = np.arange(1, 23)
    for i, alpha in enumerate(alphas):
        Ne = bootstrap_single_run(chs, ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict, Ninit, Tmax, \
                                  minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, alpha=alpha, beta=beta,\
                                    FP=FP, R=R, POWER=POWER)
        histograms, binMidpoint, G = prepare_input(ibds_by_chr, chs, ch_len_dict, nSamples.keys(), \
                                        minL=minL_calc, maxL=maxL_calc, step=step)
        bins_infer = np.arange(minL_infer, maxL_infer+step, step)
        binMidpoint_infer = (bins_infer[1:] + bins_infer[:-1])/2
        s = np.where(np.isclose(binMidpoint, binMidpoint_infer[0]))[0][0]
        e = np.where(np.isclose(binMidpoint, binMidpoint_infer[-1]))[0][0]
        devstat, _ = singlePop_MultiTP_given_vecNe_negLoglik(Ne, histograms, binMidpoint, G, gaps, nSamples, Tmax, 0, 0, timeBoundDict,\
                                                                    s=s, e=e, FP=FP, R=R, POWER=POWER)
        devstats[i] = devstat
        Nes.append(Ne)
    plt.plot(alphas, devstats, marker='o')
    plt.xlabel('alpha')
    plt.ylabel('deviance statistics')
    plt.xscale('log')

    delimiter = '.'
    plt.savefig(os.path.join(outfolder, delimiter.join([s for s in [prefix, "hyperparam_devstat.pdf"] if s.strip()])), dpi=300)
    # save the numpy array loglik to a file
    np.save(os.path.join(outfolder, delimiter.join([s for s in [prefix, 'hyperparam_devstat.npy'] if s.strip()])), devstats)
    if history:
       pickle.dump(Nes, open(os.path.join(outfolder, \
                       delimiter.join([s for s in [prefix, "hyperparam_devstat.Nes.pickle"] if s.strip()])), 'wb'))
    
    ########### now, determine the best hyperparam ################
    # first, find the largest alpha that gives a loglikelihood within 1% of the maximum
    min_dev_stat = np.min(devstats)
    min_alpha_index = np.argmin(devstats)
    print(f'min_dev_stat={min_dev_stat}, max_alpha_index={min_alpha_index}, alpha_with_minDev={alphas[min_alpha_index]}')
    # find the largest alpha that within 1.1 from min_dev_stat
    opt_alpha_index = np.max(np.where(devstats <= min_dev_stat + 1.2)[0])
    opt_alpha = alphas[opt_alpha_index]
    print(f'route 1: opt_alpha={opt_alpha}')

    return max(2500, opt_alpha)

###################################################################################################################

def get_ibdHistogram_for_each_train_val_split(df_ibd, sampleCluster2id, k_fold,
                                            minL_calc=2.0, maxL_calc=22, step=0.1):
    # this is a map from kfold validation id (from 0 to kfold -1) to a tuple of four dictionaries:
    # ibdHistogram_train, ibdHistogram_val, nHaplotypePairs_train, nHaplotypePairs_val
    # each dictionary maps a pair of sample clusters to what their naming suggests
    ibdHistogram_for_each_train_val_split = {}
    # first, for each pairs of time points, split pairs of samplesby k_fold
    sampleClusterPairs = list(itertools.combinations_with_replacement(sampleCluster2id.keys(), 2))
    sampleClusterPairs_2_ID_pair_split = {}
    for sampleClusterPair in sampleClusterPairs:
        if sampleClusterPair[0] == sampleClusterPair[1]:
            pairs = list(itertools.combinations(sampleCluster2id[sampleClusterPair[0]], 2))
        else:
            pairs = list(itertools.product(sampleCluster2id[sampleClusterPair[0]], sampleCluster2id[sampleClusterPair[1]]))
        np.random.shuffle(pairs) # the sequence is modified IN-PLACE
        sampleClusterPairs_2_ID_pair_split[sampleClusterPair] = np.array_split(pairs, k_fold)        

    bins = np.arange(minL_calc, maxL_calc+step, step)
    time_accu = 0.0
    for i in range(k_fold):
        ibdHistogram_train = {}
        ibdHistogram_val = {}
        nHaplotypePairs_train = defaultdict(lambda: 0)
        nHaplotypePairs_val = defaultdict(lambda: 0)
        
        # select pairs of samples to use for the validation set
        time1 = time.time()
        for sampleClusterPair in sampleClusterPairs:
            validation_pairs = sampleClusterPairs_2_ID_pair_split[sampleClusterPair][i]
            nHaplotypePairs_val[sampleClusterPair] = 4*len(validation_pairs)
            dfs_val = []
            for pair in validation_pairs:
                dfs_val.append(df_ibd[((df_ibd['iid1'] == pair[0]) & (df_ibd['iid2'] == pair[1]))
                                      | ((df_ibd['iid1'] == pair[1]) & (df_ibd['iid2'] == pair[0]))])
            df_val = pd.concat(dfs_val)
            hist, _ = np.histogram(100*df_val['lengthM'].values, bins=bins)
            ibdHistogram_val[sampleClusterPair] = hist

            dfs_train = []
            for j in range(k_fold):
                if j != i:
                    train_pairs = sampleClusterPairs_2_ID_pair_split[sampleClusterPair][j]
                    nHaplotypePairs_train[sampleClusterPair] += 4*len(train_pairs)
                    for pair in train_pairs:
                        dfs_train.append(df_ibd[((df_ibd['iid1'] == pair[0]) & (df_ibd['iid2'] == pair[1]))
                                      | ((df_ibd['iid1'] == pair[1]) & (df_ibd['iid2'] == pair[0]))])
                
            df_train = pd.concat(dfs_train)
            hist, _ = np.histogram(100*df_train['lengthM'].values, bins=bins)
            ibdHistogram_train[sampleClusterPair] = hist
        time_accu += time.time() - time1
        # convert defaultdict to an ordinary dict
        nHaplotypePairs_train = dict(nHaplotypePairs_train)
        nHaplotypePairs_val = dict(nHaplotypePairs_val)
        ibdHistogram_for_each_train_val_split[i] = (ibdHistogram_train, ibdHistogram_val, nHaplotypePairs_train, nHaplotypePairs_val)
    
    print(f'time spent on subsetting dataframe: {time_accu:.3f}s')
    return ibdHistogram_for_each_train_val_split

def getLoglik_kfold(ibdHistogram_for_each_train_val_split, dates, chrlens, timeBoundDict, alpha, k_fold, Ninit=500, Tmax=100,
                       minL_calc=2.0, maxL_calc=22, minL_infer=6.0, maxL_infer=20.0, step=0.1, 
                       beta=250, FP=None, R=None, POWER=None, method='l2'):

    logliks = []
    devstats = []
    bins = np.arange(minL_calc, maxL_calc+step, step)
    binMidpoint = (bins[1:] + bins[:-1])/2
    bins_infer = np.arange(minL_infer, maxL_infer+step, step)
    binMidpoint_infer = (bins_infer[1:] + bins_infer[:-1])/2
    s = np.where(np.isclose(binMidpoint, binMidpoint_infer[0]))[0][0]
    e = np.where(np.isclose(binMidpoint, binMidpoint_infer[-1]))[0][0]
    for i in range(k_fold):
        ibdHistogram_train, ibdHistogram_val, nHaplotypePairs_train, nHaplotypePairs_val = ibdHistogram_for_each_train_val_split[i]
        # now, we have the training and validation set, we can start the inference
        Ne = bootstrap_single_run(ibdHistogram_train, chrlens, dates, nHaplotypePairs_train, timeBoundDict, 
                Ninit=Ninit, Tmax=Tmax, minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer,
                step=step, alpha=alpha, beta=beta, FP=FP, R=R, POWER=POWER, method=method)
        # compute the loglikelihood of the validation set
        loglik, _= singlePop_MultiTP_given_vecNe_negLoglik(Ne, ibdHistogram_val, binMidpoint, chrlens, dates, 
                                            nHaplotypePairs_val, Tmax, 0, 0, timeBoundDict,
                                            s=s, e=e, FP=FP, R=R, POWER=POWER)
        devstat = singlePop_MultiTP_given_vecNe_DevStat(Ne, ibdHistogram_val, binMidpoint, chrlens, dates,
                                            nHaplotypePairs_val, Tmax, timeBoundDict,
                                            s=s, e=e, FP=FP, R=R, POWER=POWER)
        logliks.append(-loglik)
        devstats.append(devstat)
    return logliks, devstats

def getLoglik_kfold_wrapper(args):
    return getLoglik_kfold(*args)


def hyperparam_kfold(df_ibd, sampleCluster2id, dates, chr_lens, timeBoundDict, kfold=5, Ninit=500, Tmax=100, \
                    minL_calc=2.0, maxL_calc=22, minL_infer=6.0, maxL_infer=20.0,
                    FP=None, R=None, POWER=None, step=0.1, beta=250, prefix="", outfolder="", save=False,
                    parallel=True, nprocess=10, method='l2'):
    alphas = np.logspace(np.log10(50), 7, 30)
    logliks = np.zeros((len(alphas), kfold))
    devstats = np.zeros((len(alphas), kfold))
    df_ibd = df_ibd[df_ibd['lengthM'] >= minL_infer/100] # to speed-up dataframe subsetting
    ibdHistogram_for_each_train_val_split = get_ibdHistogram_for_each_train_val_split(df_ibd, sampleCluster2id, kfold, minL_calc, maxL_calc, step)
    if not parallel:
        for i, alpha in tqdm(enumerate(alphas)):
            logliks_on_val, devstats_on_val = getLoglik_kfold(ibdHistogram_for_each_train_val_split, dates, chr_lens, timeBoundDict, alpha, kfold, Ninit, Tmax, \
                                  minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, 
                                  beta=beta, FP=FP, R=R, POWER=POWER, method=method)
            print(f'alpha={alpha:.3f}, avg_loglik_on_val={np.mean(logliks_on_val):.3f}, avg_devstat_on_val={np.mean(devstats_on_val):.3f}', flush=True)
            logliks[i] = logliks_on_val
            devstats[i] = devstats_on_val
    else:
        params = [[ibdHistogram_for_each_train_val_split, dates, chr_lens, timeBoundDict, alpha, kfold, Ninit, Tmax, 
                        minL_calc, maxL_calc, minL_infer, maxL_infer, step, 
                        beta, FP, R, POWER, method] for alpha in alphas]
        with mp.Pool(processes = nprocess) as pool:
            results = list(tqdm(pool.imap(getLoglik_kfold_wrapper, params), total=len(alphas)))
        for i, (logliks_on_val, devstats_on_val) in enumerate(results):
            print(f'alpha={alphas[i]:.3f}, avg_loglik_on_val={np.mean(logliks_on_val):.3f}, avg_devstat_on_val={np.mean(devstats_on_val):.3f}', flush=True)
            logliks[i] = logliks_on_val
            devstats[i] = devstats_on_val
    if save:
        cv_record = {'logliks':logliks, 'devstats':devstats, 'alphas':alphas}
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        pickle.dump(cv_record, open(os.path.join(outfolder, 
                       '.'.join([s for s in [prefix, "cv.trace.pickle"] if s.strip()])), 'wb'))
        
    devstats_mean = np.mean(devstats, axis=1)
    min_deviance_stat = np.min(devstats_mean)
    opt_alpha_index = np.where(devstats_mean <= min_deviance_stat + 1.0)[0][-1]
    print(f'opt alpha: {alphas[opt_alpha_index]:.3f}')
    return alphas[opt_alpha_index]

