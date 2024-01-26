import numpy as np
import itertools
from collections import defaultdict
from scipy.optimize import minimize
from scipy.ndimage import shift
from scipy.stats import norm
from analytic import singlePop_2tp_given_Ne_negLoglik, \
    singlePop_2tp_given_vecNe_negLoglik_noPenalty,\
    singlePop_2tp_given_vecNe_DevStat_noPenalty
from ts_utility import multi_run
from plot import plot_pairwise_fitting, plot_pairwise_TMRCA, plotPosteriorTMRCA, plot2PopIMfit
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

ch_len_dict = {1:286.279, 2:268.840, 3:223.361, 4:214.688, 5:204.089, 6:192.040, 7:187.221, 8:168.003, 9:166.359, \
        10:181.144, 11:158.219, 12:174.679, 13:125.706, 14:120.203, 15:141.860, 16:134.038, 17:128.491, 18:117.709, \
        19:107.734, 20:108.267, 21:62.786, 22:74.110}


def singlePop_MultiTP_given_Ne_negLoglik(Ne, histograms, binMidpoint, chrlens, dates, nHaplotypePairs, timeBoundDict, s=0, e=-1, FP=None, R=None, POWER=None):
    accu = 0
    for (clusterID1, clusterID2), nHaplotypePair in nHaplotypePairs.items():
        if nHaplotypePair == 0:
            continue
        clusterID1, clusterID2 = min(clusterID1, clusterID2), max(clusterID1, clusterID2)
        accu += singlePop_2tp_given_Ne_negLoglik(Ne, histograms[(clusterID1, clusterID2)], 
                binMidpoint, chrlens, dates[clusterID1], dates[clusterID2], nHaplotypePair,
                [(timeBoundDict[clusterID1]), (timeBoundDict[clusterID2])], s, e, FP, R, POWER)
    return accu

def inferConstNe_singlePop_MultiTP(histograms, binMidpoint, dates, nHaplotypePairs, Ninit, chrlens, timeBoundDict, s=0, e=-1, FP=None, R=None, POWER=None):
    # given the observed ibd segments, estimate the Ne assuming a constant effective Ne
    # dates: dictionary, where the key is the sampling cluter index and the value is the sampling time (in generations ago)
    # nSamples: dictionary, where the key is the sampling cluster index and the value is the number of diploid samples within each cluster

    kargs = (histograms, binMidpoint, chrlens, dates, nHaplotypePairs, timeBoundDict, s, e, FP, R, POWER)
    res = minimize(singlePop_MultiTP_given_Ne_negLoglik, Ninit, args=kargs, method='L-BFGS-B', bounds=[(10, 5e6)])
    return res.x[0]

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
        accu_, grad_ = singlePop_2tp_given_vecNe_negLoglik_noPenalty(Ne, histograms[(clusterID1, clusterID2)],
            binMidpoint, chrlens, dates[clusterID1], dates[clusterID2], Tmax, nHaplotypePair, 
            [timeBoundDict[clusterID1], timeBoundDict[clusterID2]], s=s, e=e, FP=FP, R=R, POWER=POWER, tail=tail)
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

def bootstrap_single_run(ch_ids, ibds_by_chr, dates, nHaplotypePairs, ch_len_dict, 
        timeBoundDict, Ninit=500, Tmax=100,
        minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0,
        step=0.1, alpha=1e-6, beta=1e-4, method='l2', FP=None, R=None, POWER=None, verbose=False):
    # perform one bootstrap resampling

    ###################### First, infer a const Ne ################################
    histograms, binMidpoint, chrlens = prepare_input(ibds_by_chr, ch_ids, ch_len_dict, dates.keys(), minL=minL_calc, maxL=maxL_calc, step=step)
    bins_infer = np.arange(minL_infer, maxL_infer+step, step)
    binMidpoint_infer = (bins_infer[1:] + bins_infer[:-1])/2
    s = np.where(np.isclose(binMidpoint, binMidpoint_infer[0]))[0][0]
    e = np.where(np.isclose(binMidpoint, binMidpoint_infer[-1]))[0][0]
    if (not FP is None) and (not R is None) and (not POWER is None):
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
        ibds_, nHaplotypePairs_ = {}, {}
        ibds_[(clusterID, clusterID)] = ibds_by_chr[(clusterID, clusterID)]
        nHaplotypePairs_[(clusterID, clusterID)] = nHaplotypePairs[(clusterID, clusterID)]
        histograms_, binMidpoint_, chrlens_ = prepare_input(ibds_, np.arange(1,23), ch_len_dict, [clusterID], minL=minL_calc, maxL=maxL_calc, step=step)
        constNe = inferConstNe_singlePop_MultiTP(histograms_, binMidpoint_, {clusterID:0}, nHaplotypePairs_, Ninit, chrlens_, timeBoundDict, s, e, FP, R, POWER)
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

def bootstrap(func, resample_chrs, nprocess, *args):
    """
    fullResults: whether to return the full results of individual bootstrap resampling
    """
    # perform bootstrap resampling, and then return the 95% CI of the estimated Ne trajectory
    time1 = time.time()
    args = list(args)
    params = [[resample_chr, *args] for resample_chr in resample_chrs]
    results = multi_run(func, params, processes=nprocess, output=True)
    results_aggregate = np.zeros((len(results), len(results[0])))
    for i in range(len(results)):
        results_aggregate[i] = results[i]
    results_aggregate_sorted = np.sort(results_aggregate, axis=0)
    print(f'bootstrap takes {time.time() - time1:.3f}s')

    index = int(2.5/(100/len(resample_chrs)))
    return results_aggregate_sorted[index-1], results_aggregate_sorted[-index]

    

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


def inferVecNe_singlePop_MultiTP(ibds_by_chr, dates, nSamples, ch_len_dict, timeBoundDict=None, Ninit=500, Tmax=100, \
        minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0,
        step=0.1, alpha=2500, beta=250, method='l2', FP=None, R=None, POWER=None, nprocess=6, plot=False, prefix="", \
        verbose=True, doBootstrap=True, autoHyperParam=False, outFolder='.'):
    """
    Parameter
    ---------
    timeBound: dict
        A dictionary specifying the dating uncertainty of each of the sampling cluster. For example, 1:(-2,3) means that the dating of the sample cluster 1
        is distributed from 2 generations more recent, to 3 generations older than the time point given in the parameter gaps. We assume a uniform distribution across this range.
    
    """
    if ((FP is None) or (R is None) or (POWER is None)) and (minL_calc != minL_infer or maxL_calc != maxL_infer):
        warnings.warn('Error model not provided... Setting the length range used for calculation and inference to be the same.')
        minL_calc = minL_infer
        maxL_calc = maxL_infer

    # estimate Ne using the original data
    ch_ids = [k for k, v in ch_len_dict.items()]
    if not timeBoundDict:
        timeBoundDict = {}
        for id, _ in nSamples.items():
            timeBoundDict[id] = (0, 0)
    assert(np.min(list(dates.values())) == 0) # the most recent sample cluster must be labeled as generation 0
    # calculate # of haplotype pairs for each pairs of sample clusters
    nHaplotypePairs = {}
    for clusterID1, clusterID2 in itertools.combinations_with_replacement(dates.keys(), 2):
        clusterID1, clusterID2 = min(clusterID1, clusterID2), max(clusterID1, clusterID2)
        if clusterID1 == clusterID2:
            nHaplotypePairs[(clusterID1, clusterID2)] = 2*nSamples[clusterID1]*(2*nSamples[clusterID1]-2)/2
        else:
            nHaplotypePairs[(clusterID1, clusterID2)] = 2*nSamples[clusterID1]*2*nSamples[clusterID2]
    print(f'number of haplotype pairs: {nHaplotypePairs}')
    #### do hyperparameter search if needed
    if autoHyperParam:
        pass
        # alpha = hyperparam_opt_DevStat(ibds_by_chr, dates, nSamples, ch_len_dict, timeBoundDict, Ninit=Ninit, Tmax=Tmax, \
        #             minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer,
        #             step=step, beta=beta, weighted=weighted, prefix=prefix, outfolder=outFolder, history=True)
        # print(f'overwrite alpha to be {alpha}, determined by automatic hyperparameter search.')

    Ne = bootstrap_single_run(ch_ids, ibds_by_chr, dates, nHaplotypePairs, ch_len_dict, timeBoundDict, Ninit=Ninit, Tmax=Tmax, \
        minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, \
        step=step, alpha=alpha, beta=beta, method=method, FP=FP, R=R, POWER=POWER, verbose=verbose)
    if plot:
        if len(prefix) > 0:
            plot_prefix = "pairwise." + prefix
        else:
            plot_prefix = "pairwise"
        min_plot = math.floor(min(itertools.chain.from_iterable(ibdlist for dict in ibds_by_chr.values() for ibdlist in dict.values())))
        print(f'minL for plotting: {min_plot}')
        plot_pairwise_fitting(ibds_by_chr, dates, nHaplotypePairs, ch_len_dict, Ne, outFolder, timeBoundDict, prefix=plot_prefix, \
            minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL=maxL_infer, step=step, minL_plot=min_plot, FP=FP, R=R, POWER=POWER)
        plot_pairwise_TMRCA(dates, Ne, Tmax, outFolder, prefix=plot_prefix, minL=minL_infer, maxL=maxL_infer, step=0.25)

    if doBootstrap:
        # start bootstrapping
        nresample = 200
        ch_ids = [k for k, v in ch_len_dict.items()]
        resample_chrs = [ np.random.choice(ch_ids, size=len(ch_ids)) for i in range(nresample)]
        bootstrap_results = np.full((nresample, len(Ne)), np.nan)
        time1 = time.time()
        for i in prange(nresample):
            bootstrap_results[i] = bootstrap_single_run(resample_chrs[i], ibds_by_chr, 
                dates, nHaplotypePairs, ch_len_dict, timeBoundDict, Ninit=Ninit, Tmax=Tmax, 
                minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer,
                step=step, alpha=alpha, beta=beta, method=method, FP=FP, R=R, POWER=POWER, verbose=False)
        bootstrap_results_sorted = np.sort(bootstrap_results, axis=0)
        print(f'bootstrap takes {time.time() - time1:.3f}s')
        # lowCI, upCI = bootstrap(bootstrap_single_run, resample_chrs, nprocess, ibds_by_chr, dates, nHaplotypePairs, ch_len_dict, timeBoundDict, Ninit, Tmax, \
        #     minL_calc, maxL_calc, minL_infer, maxL_infer, step, alpha, beta, method, FP, R, POWER, False)
        # create an empty pandas dataframe with three columns named Ne, lowCI, upCI
        df = pd.DataFrame(columns=['Generations', 'Ne', 'lowCI', 'upCI'])
        df['Generations'] = np.arange(1, len(Ne)+1)
        df['Ne'] = Ne
        df['lowCI'] = bootstrap_results_sorted[int(0.025*nresample)]
        df['upCI'] = bootstrap_results_sorted[int(0.975*nresample)]
    else:
        df = pd.DataFrame(columns=['Generatioins', 'Ne'])
        df['Generations'] = np.arange(1, len(Ne)+1)
        df['Ne'] = Ne
    return df


def inferVecNe_singlePop_MultiTP_withMask(path2IBD, path2ChrDelimiter, path2mask=None, nSamples=-1, path2SampleAge=None, \
        Ninit=500, Tmax=100, minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, step=0.1, alpha=2500, beta=0, method='l2', \
        FP=None, R=None, POWER=None, generation_time=29, minSample=10, maxSample=np.inf, merge_level=5, \
        prefix="", doBootstrap=True, autoHyperParam=False, outFolder='.', run=True):
    ###### read in a list of samples with their mean BP. If no such info is provided, assume all samples are taken at the same time
    if path2SampleAge:
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
        gaps = {i: merge_level*i for i in range(0, maxBinIndex+1)}
        nSamples = {i: sum(v == i for _, v in sampleAgeBinDict.items()) for i in range(0, maxBinIndex+1)}

        exSampleCluster = set()
        for i in range(0, maxBinIndex+1):
            n = nSamples[i]
            if n < minSample or n > maxSample:
                exSampleCluster.add(i)
            print(f'{nSamples[i]} samples in age bin {i}({youngest_age + i*merge_level*generation_time} - {youngest_age + (i+1)*merge_level*generation_time}BP): {[k for k, v in sampleAgeBinDict.items() if v == i]}')
        
        for i in exSampleCluster:
            del nSamples[i]
            del gaps[i]
        print(f'Bins {[i for i, _ in gaps.items()]} will be used for inference.')
        
        timeOffset = np.min(list(gaps.values()))
        for i in gaps.keys():
            gaps[i] -= timeOffset
        
    else:
        gaps = {0:0}
        warnings.warn('No sample age list is provided, assume all samples are taken at the same time...')
        if nSamples == -1:
            warnings.warn('Number of samples not provided, assuming all samples occur in the IBD file...')
        else:
            nSamples = {0:nSamples}
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
        ibds_by_chr = defaultdict(dict)
        with open(path2IBD) as f:
            for line in f:
                iid1, iid2, ch, ibd_start, ibd_end, length = line.strip().split()
                ch, ibd_start, ibd_end, length = int(ch), float(ibd_start), float(ibd_end), float(length)
                if sampleAgeBinDict.get(iid1) != None and sampleAgeBinDict.get(iid2) != None:
                    if path2SampleAge:
                        pop1, pop2 = min(sampleAgeBinDict[iid1], sampleAgeBinDict[iid2]), max(sampleAgeBinDict[iid1], sampleAgeBinDict[iid2])
                    else:
                        pop1 = pop2 = 0
                else:
                    continue
                for i, span in enumerate(spanlist[map[ch]:map[ch+1]]):
                    # ask whether the current segment being read has any overlap with this span of interest
                    span_start, span_end = span
                    if ibd_end <= span_start or ibd_start >= span_end:
                        continue
                    else:
                        ibd_start_masked, ibd_end_masked = max(span_start, ibd_start), min(span_end, ibd_end)
                        if not ibds_by_chr[(pop1, pop2)].get(i+map[ch]):
                            ibds_by_chr[(pop1, pop2)][i+map[ch]] = []
                        ibds_by_chr[(pop1, pop2)][i+map[ch]].append(ibd_end_masked - ibd_start_masked)
    else:
        ########## no mask file provided, thus no masking will be performed ###################
        print(f'no mask file provided, all segments will be used for inference')
        ch_len_dict = {k:v[1]-v[0] for k, v in chrDelimiter.items()}
        ibds_by_chr = defaultdict(dict)
        with open(path2IBD) as f:
            for line in f:
                iid1, iid2, ch, ibd_start, ibd_end, length = line.strip().split()
                ch, ibd_start, ibd_end, length = int(ch), float(ibd_start), float(ibd_end), float(length)
                if sampleAgeBinDict.get(iid1) != None and sampleAgeBinDict.get(iid2) != None:
                    if path2SampleAge:
                        pop1, pop2 = min(sampleAgeBinDict[iid1], sampleAgeBinDict[iid2]), max(sampleAgeBinDict[iid1], sampleAgeBinDict[iid2])
                    else:
                        pop1 = pop2 = 0
                    if not ibds_by_chr[(pop1, pop2)].get(ch):
                        ibds_by_chr[(pop1, pop2)][ch] = []
                    ibds_by_chr[(pop1, pop2)][ch].append(length)
                else:
                    continue

    filename = 'ibds_by_chr.pickle' if len(prefix) == 0 else 'ibds_by_chr.' + prefix + '.pickle'
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    pickle.dump(ibds_by_chr, open(os.path.join(outFolder, filename), 'wb'))
    ### end of preprocessing of input data, ready to start the inference
    if run:
        return inferVecNe_singlePop_MultiTP(ibds_by_chr, gaps, nSamples, ch_len_dict, Ninit=Ninit, Tmax=Tmax, \
                minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, alpha=alpha, beta=beta, \
                method=method, FP=FP, R=R, POWER=POWER, plot=True, \
                prefix=prefix, autoHyperParam=autoHyperParam, doBootstrap=doBootstrap, outFolder=outFolder)
    else:
        return ibds_by_chr, gaps, nSamples, ch_len_dict, FP, R, POWER


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

def getAvgLoglik_kfold(df_ibd, sampleCluster2id, gaps, ch_len_dict, timeBoundDict, alpha, k_fold, Ninit=500, Tmax=100,
                       minL_calc=2.0, maxL_calc=22, minL_infer=6.0, maxL_infer=20.0, step=0.1, 
                       beta=250, FP=None, R=None, POWER=None):
    logliks = []

    # first, for each pairs of time points, split pairs of samplesby k_fold
    sampleClusterPairs = [(x,x) for x in sampleCluster2id.keys()] + itertools.combinations(sampleCluster2id.keys(), 2)
    sampleClusterPairs_2_ID_pair_split = {}
    for sampleClusterPair in sampleClusterPairs:
        if sampleClusterPair[0] == sampleClusterPair[1]:
            pairs = list(itertools.combinations(sampleCluster2id[sampleClusterPair[0]], 2))
        else:
            pairs = list(itertools.product(sampleCluster2id[sampleClusterPair[0]], sampleCluster2id[sampleClusterPair[1]]))
        np.random.shuffle(pairs) # the sequence is modified IN-PLACE
        sampleClusterPairs_2_ID_pair_split[sampleClusterPair] = np.array_split(pairs, k_fold)

    for i in range(k_fold):
        ibds_train = {}
        ibds_val = {}
        # select pairs of samples to use for the validation set
        for sampleClusterPair in sampleClusterPairs:
            validation_pairs = sampleClusterPairs_2_ID_pair_split[sampleClusterPair][i]
            dfs_val = []
            for pair in validation_pairs:
                dfs_val.append(df_ibd[((df_ibd['iid1'] == pair[0]) & (df_ibd['iid2'] == pair[1]))
                                      | ((df_ibd['iid1'] == pair[1]) & (df_ibd['iid2'] == pair[0]))])
            df_val = pd.concat(dfs_val)

            dfs_train = []
            for j in range(k_fold):
                if j != i:
                    train_pairs = sampleClusterPairs_2_ID_pair_split[sampleClusterPair][j]
                    for pair in train_pairs:
                        dfs_train.append(df_ibd[((df_ibd['iid1'] == pair[0]) & (df_ibd['iid2'] == pair[1]))
                                      | ((df_ibd['iid1'] == pair[1]) & (df_ibd['iid2'] == pair[0]))])
                
            df_train = pd.concat(dfs_train)
            ibds_train[sampleClusterPair] = df_train
            ibds_val[sampleClusterPair] = df_val
        
        # now, we have the training and validation set, we can start the inference
            
        

    return np.mean(logliks)

    


def hyperparam_kfold(df_ibd, id2SampleCluster, gaps, ch_len_dict, timeBoundDict, Ninit=500, Tmax=100, \
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