import numpy as np
import itertools
from collections import defaultdict
from scipy.optimize import minimize
from scipy.ndimage import shift
from scipy.stats import norm
from analytic import singlePop_2tp_given_Ne_negLoglik, singlePop_2tp_given_vecNe_negLoglik_noPenalty, \
    twoPopIM_2tp_given_coalrate_negLoglik, twoPopIM_given_vecCoalRates_negLoglik_noPenalty
from ts_utility import multi_run
from plot import plot_pairwise_fitting, plot_pairwise_TMRCA, plotPosteriorTMRCA, plot2PopIMfit
#from AccProx import AccProx_trendFiltering_l1
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


def singlePop_MultiTP_given_Ne_negLoglik(Ne, histograms, binMidpoint, G, gaps, nSamples, timeBoundDict, s=0, e=-1, FP=None, R=None, POWER=None):
    accu = 0
    for id, nSample in nSamples.items():
        if nSample == 1:
            continue
        accu += singlePop_2tp_given_Ne_negLoglik(Ne, histograms[(id, id)], binMidpoint, G, gaps[id], gaps[id],\
            (2*nSample)*(2*nSample-2)/2, [(timeBoundDict[id]), (timeBoundDict[id])], s, e, FP, R, POWER)
    for id1, id2 in itertools.combinations(nSamples.keys(), 2):
        accu += singlePop_2tp_given_Ne_negLoglik(Ne, histograms[(min(id1, id2), max(id1, id2))], binMidpoint, \
                G, gaps[id1], gaps[id2], (2*nSamples[id1])*(2*nSamples[id2]),\
                [(timeBoundDict[id1]), (timeBoundDict[id2])], s, e, FP, R, POWER)
    return accu

def inferConstNe_singlePop_MultiTP(histograms, binMidpoint, gaps, nSamples, Ninit, chrlens, timeBoundDict, s=0, e=-1, FP=None, R=None, POWER=None):
    # given the observed ibd segments, estimate the Ne assuming a constant effective Ne
    # gaps: dictionary, where the key is the sampling cluter index and the value is the sampling time
    # nSamples: dictionary, where the key is the sampling cluster index and the value is the number of diploid samples within each cluster

    kargs = (histograms, binMidpoint, chrlens, gaps, nSamples, timeBoundDict, s, e, FP, R, POWER)
    res = minimize(singlePop_MultiTP_given_Ne_negLoglik, Ninit, args=kargs, method='L-BFGS-B', bounds=[(10, 5e6)])
    return res.x[0]


def singlePop_MultiTP_given_vecNe_negLoglik_noPenalty(Ne, histograms, binMidpoint, G, gaps, nSamples, Tmax, timeBoundDict, \
                                            s=0, e=-1, FP=None, R=None, POWER=None, tail=False):
    accu = 0

    ################################ unify computation for both within and between sample cluster ############################
    SampleClusterID = list(nSamples.keys())
    sampleClusterCombos = list(itertools.combinations_with_replacement(SampleClusterID, 2))
    for i in prange(len(sampleClusterCombos)):
        id1, id2 = sampleClusterCombos[i]
        npairs = (2*nSamples[id1])*(2*nSamples[id2]) if id1 != id2 else (2*nSamples[id1])*(2*nSamples[id2]-2)/2
        accu_, _ = singlePop_2tp_given_vecNe_negLoglik_noPenalty(Ne, histograms[(min(id1, id2), max(id1, id2))], \
            binMidpoint, G, gaps[id1], gaps[id2], Tmax, npairs, [timeBoundDict[id1], timeBoundDict[id2]], 
            s=s, e=e, FP=FP, R=R, POWER=POWER, tail=tail)
        accu += accu_
    return accu

def singlePop_MultiTP_given_vecNe_negLoglik(Ne, histograms, binMidpoint, G, gaps, nSamples, Tmax, alpha, beta, timeBoundDict, Nconst, s=0, e=-1, FP=None, R=None, POWER=None, tail=False):
    # return the negative loglikelihood of Ne, plus the penalty term
    # G: a vector of chromosome length
    # gaps: dictionary, where the key is the sampling cluter index and the value is the sampling time
    # nSamples: dictionary, where the key is the sampling cluster index and the value is the number of diploid samples within each cluster
    accu = 0
    grad = np.zeros_like(Ne)

    ################################ unify computation for both within and between sample cluster ############################
    SampleClusterID = list(nSamples.keys())
    sampleClusterCombos = list(itertools.combinations_with_replacement(SampleClusterID, 2))
    for i in prange(len(sampleClusterCombos)):
        id1, id2 = sampleClusterCombos[i]
        npairs = (2*nSamples[id1])*(2*nSamples[id2]) if id1 != id2 else (2*nSamples[id1])*(2*nSamples[id2]-2)/2
        accu_, grad_ = singlePop_2tp_given_vecNe_negLoglik_noPenalty(Ne, histograms[(min(id1, id2), max(id1, id2))], \
            binMidpoint, G, gaps[id1], gaps[id2], Tmax, npairs, [timeBoundDict[id1], timeBoundDict[id2]], 
            s=s, e=e, FP=FP, R=R, POWER=POWER, tail=tail)
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

def bootstrap_single_run(ch_ids, ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict, Ninit=500, Tmax=100, \
        minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, \
        step=0.1, alpha=1e-6, beta=1e-4, method='l2', FP=None, R=None, POWER=None, verbose=False):
    # perform one bootstrap resampling

    ###################### First, infer a const Ne ################################
    histograms, binMidpoint, chrlens = prepare_input(ibds_by_chr, ch_ids, ch_len_dict, nSamples.keys(), minL=minL_calc, maxL=maxL_calc, step=step)
    bins_infer = np.arange(minL_infer, maxL_infer+step, step)
    binMidpoint_infer = (bins_infer[1:] + bins_infer[:-1])/2
    s = np.where(np.isclose(binMidpoint, binMidpoint_infer[0]))[0][0]
    e = np.where(np.isclose(binMidpoint, binMidpoint_infer[-1]))[0][0]
    if (not FP is None) and (not R is None) and (not POWER is None):
        assert(len(FP) == len(binMidpoint))
        assert(len(POWER) == len(binMidpoint))
        assert(R.shape[0] == R.shape[1] == len(binMidpoint))
    Nconst = inferConstNe_singlePop_MultiTP(histograms, binMidpoint, gaps, nSamples, Ninit, chrlens, timeBoundDict, s, e, FP, R, POWER)
    if verbose:
        print(f'estimated constant Ne: {Nconst}')
    ################################################################################
    histograms, binMidpoint, chrlens = prepare_input(ibds_by_chr, ch_ids, ch_len_dict, nSamples.keys(), minL=minL_calc, maxL=maxL_calc, step=step)
    t_min = np.min(list(gaps.values()))
    t_max = np.max(list(gaps.values()))
    i_min = -1
    i_max = -1
    for k, v in gaps.items():
        if v == t_min:
            i_min = k
        if v == t_max:
            i_max = k
    assert(i_min != -1)
    assert(i_max != -1)
    t_min += timeBoundDict[i_min][0] # here t_min is either 0 or negative
    t_max += timeBoundDict[i_max][1]
    ### shift each sample cluster's time by |t_min|
    gaps_adjusted = {}
    for k, v in gaps.items():
        gaps_adjusted[k] = v + abs(t_min)

    #NinitVec = np.full(Tmax + (t_max - t_min), Nconst)
    #NinitVec = np.exp(np.random.normal(np.log(Nconst), np.log(Nconst)/25, Tmax + (t_max - t_min)))
    NinitVec = np.random.normal(Nconst, Nconst/25, Tmax + (t_max - t_min))
    # NinitVec = np.full(Tmax + (t_max - t_min), np.nan)
    for id in nSamples.keys():
        ibds_, nSamples_ = {}, {}
        ibds_[(id,id)] = ibds_by_chr[(id,id)]
        nSamples_[id] = nSamples[id]
        histograms_, binMidpoint_, G_ = prepare_input(ibds_, np.arange(1,23), ch_len_dict, nSamples_.keys(), minL=minL_calc, maxL=maxL_calc, step=step)
        constNe = inferConstNe_singlePop_MultiTP(histograms_, binMidpoint_, {id:0}, nSamples_, Ninit, G_, timeBoundDict, s, e, FP, R, POWER)
        if verbose:
            print(f'constNe for {id} is {constNe}')
    #     NinitVec[gaps[id]] = constNe
    # timeAnchor = np.sort(list(gaps.values()))
    # if len(timeAnchor) == 1:
    #     Nconst = NinitVec[timeAnchor[0]]
    #     NinitVec = np.random.normal(Nconst, Nconst/20, Tmax + (t_max - t_min))
    # else:
    #     for i in range(len(timeAnchor)-1):
    #         # note the direction of time
    #         end, start = timeAnchor[i], timeAnchor[i+1]
    #         Nend, Nstart = NinitVec[end], NinitVec[start]
    #         # fit an exponential growth in between Nend and Nstart
    #         if i == 0:
    #             end = 0 # where there is timeBound, timeAnchor[0] may not be 0; so we set it to be 0 to extend the exponential growth to the very beginning
    #         t = np.arange(start-end)[::-1]
    #         NinitVec[end:start] = Nstart*np.exp((t/(start-end))*np.log(Nend/Nstart))
    #         if i == len(timeAnchor)-2:
    #             NinitVec[start:] = np.random.normal(NinitVec[start], NinitVec[start]/20, Tmax + (t_max - t_min) - start)
    # # if verbose:
    # #     print(f'NinitVec: {NinitVec}')
    # assert(~np.isnan(NinitVec).any())  

    kargs = (histograms, binMidpoint, chrlens, gaps_adjusted, nSamples, Tmax, alpha, beta, timeBoundDict, Nconst, s, e+1, FP, R, POWER)  
    
    if method == 'l2':
        t1 = time.time()
        res = minimize(singlePop_MultiTP_given_vecNe_negLoglik, NinitVec, \
                args=kargs, method='L-BFGS-B', jac=True, bounds=[(10, 5e6) for i in range(Tmax + (t_max - t_min))], \
                options={'maxfun': 50000, 'maxiter':50000, 'gtol':1e-8, 'ftol':1e-8})
        if verbose:
            print(res)
            print(f'elapsed time for optimization: {time.time() - t1}')
        return res.x
    elif method == 'l1':
        raise RuntimeError('Unsupported Method')
        # res = AccProx_trendFiltering_l1(NinitVec, *kargs)
        # if verbose:
        #     print(res)
        # return res
    else:
        raise RuntimeError('Unsupported Method')

def bootstrap(func, resample_chrs, nprocess, fullResults, *args):
    """
    fullResults: whether to return the full results of individual bootstrap resampling
    """
    # perform bootstrap resampling, and then return the 95% CI of the estimated Ne trajectory
    args = list(args)
    params = [[resample_chr, *args] for resample_chr in resample_chrs]
    results = multi_run(func, params, processes=nprocess, output=True)
    results_aggregate = np.zeros((len(results), len(results[0])))
    for i in range(len(results)):
        results_aggregate[i] = results[i]
    results_aggregate_sorted = np.sort(results_aggregate, axis=0)

    index = int(2.5/(100/len(resample_chrs)))
    if not fullResults:
        return results_aggregate_sorted[index-1], results_aggregate_sorted[-index]
    else:
        return results_aggregate_sorted[index-1], results_aggregate_sorted[-index], results_aggregate

    

def prepare_input(ibds_by_chr, ch_ids, ch_len_dict, sampleIDs, minL=4.0, maxL=20.0, step=0.1):
    ibds = defaultdict(list)
    for pair, ibd_dict_pair in ibds_by_chr.items():
        for ch in ch_ids:
            if ibd_dict_pair.get(ch):
                ibds[pair].extend(ibd_dict_pair[ch])

    bins = np.arange(minL, maxL+step, step=step)
    binMidpoint = (bins[1:] + bins[:-1])/2
    histograms = {}
    for id1, id2 in itertools.combinations_with_replacement(sampleIDs, 2):
        histogram, _ = np.histogram(ibds[(min(id1, id2), max(id1, id2))], bins=bins)
        histograms[(min(id1, id2), max(id1, id2))] = histogram
    
    chrlens = [ch_len_dict[ch] for ch in ch_ids]
    return histograms, binMidpoint, np.array(chrlens)


def inferVecNe_singlePop_MultiTP(ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict=None, Ninit=500, Tmax=100, \
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
    assert(np.min(list(gaps.values())) == 0) # the most recent sample cluster must be labeled as generation 0

    #### do hyperparameter search if needed
    if autoHyperParam:
        hyperparam_opt2(ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict, Ninit=Ninit, Tmax=Tmax, \
                    minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer,
                    step=step, beta=beta, prefix=prefix, outfolder=outFolder, history=True)

    Ne = bootstrap_single_run(ch_ids, ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict, Ninit=Ninit, Tmax=Tmax, \
        minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, \
        step=step, alpha=alpha, beta=beta, method=method, FP=FP, R=R, POWER=POWER, verbose=verbose)
    if plot:
        if len(prefix) > 0:
            plot_prefix = "pairwise." + prefix
        else:
            plot_prefix = "pairwise"
        min_plot = math.floor(min(itertools.chain.from_iterable(ibdlist for dict in ibds_by_chr.values() for ibdlist in dict.values())))
        print(f'minL for plotting: {min_plot}')
        plot_pairwise_fitting(ibds_by_chr, gaps, nSamples, ch_len_dict, Ne, outFolder, timeBoundDict, prefix=plot_prefix, \
            minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL=maxL_infer, step=step, minL_plot=min_plot, FP=FP, R=R, POWER=POWER)
        #plot_pairwise_TMRCA(gaps, Ne, Tmax, outFolder, prefix=plot_prefix, minL=minL_infer, maxL=maxL_infer, step=0.25)

    if doBootstrap:
        # start bootstrapping
        nresample = 200
        ch_ids = [k for k, v in ch_len_dict.items()]
        resample_chrs = [ np.random.choice(ch_ids, size=len(ch_ids)) for i in range(nresample)]
        # create an empty pandas dataframe with three columns named Ne, lowCI, upCI
        df = pd.DataFrame(columns=['Generations', 'Ne', 'lowCI', 'upCI'])
        lowCI, upCI = bootstrap(bootstrap_single_run, resample_chrs, nprocess, False, ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict, Ninit, Tmax, \
            minL_calc, maxL_calc, minL_infer, maxL_infer, step, alpha, beta, method, FP, R, POWER, False)
        df['Generations'] = np.arange(1, len(Ne)+1)
        df['Ne'] = Ne
        df['lowCI'] = lowCI
        df['upCI'] = upCI
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
    
        # save this ibd dictionary for future reference
        # if len(prefix) == 0:
        #     outDest = path2IBD +'.masked.pickle'
        # else:
        #     outDest = path2IBD + "." + prefix + '.masked.pickle'
        # pickle.dump(ibds_by_chr, open(outDest, 'wb'))
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
    pickle.dump(ibds_by_chr, open(os.path.join(outFolder, filename), 'wb'))
    ### end of preprocessing of input data, ready to start the inference
    if run:
        return inferVecNe_singlePop_MultiTP(ibds_by_chr, gaps, nSamples, ch_len_dict, Ninit=Ninit, Tmax=Tmax, \
                minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, alpha=alpha, beta=beta, \
                method=method, FP=FP, R=R, POWER=POWER, plot=True, \
                prefix=prefix, autoHyperParam=autoHyperParam, doBootstrap=doBootstrap, outFolder=outFolder)
    else:
        return ibds_by_chr, gaps, nSamples, ch_len_dict, FP, R, POWER


def kfold_validation(ibds_by_chr, gaps, nSamples, timeBoundDict, alpha, k_fold, Ninit=500, Tmax=100, \
                     minL_calc=2.0, maxL_calc=22, minL_infer=4.0, maxL_infer=20.0, step=0.1, beta=250):
    # return the likelihood averaged over the 5 validation sets with the given alpha
    results = np.zeros(len(k_fold))
    for i, validation_set in enumerate(k_fold):
        train_set = [i for i in np.arange(1, 23) if i not in validation_set]
        Ne = bootstrap_single_run(train_set, ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict, Ninit, Tmax, \
                                  minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, alpha=alpha, beta=beta)
        histograms, binMidpoint, G = prepare_input(ibds_by_chr, validation_set, ch_len_dict, nSamples.keys(), \
                                        minL=minL_calc, maxL=maxL_calc, step=step)
        # since here we only want the likelihood, not the penalty term, we can safely set Nconst, alpha, beta all to 0
        loglike, _ = singlePop_MultiTP_given_vecNe_negLoglik(Ne, histograms, binMidpoint, G, gaps, nSamples, Tmax, 0, 0, timeBoundDict, 0)
        results[i] = -loglike # the function returns the negative of loglikelihood, so we need to reverse the sign
    print(f'{alpha}: {np.mean(results)}')
    return np.mean(results) 


def hyperparam_opt(ibds_by_chr, gaps, nSamples, timeBoundDict, Ninit=500, Tmax=100, minL_calc=2.0, maxL_calc=22, minL_infer=6.0, maxL_infer=20.0,
                   step=0.1, beta=250, nprocess=6, outfolder=""):
    k_fold = [[2, 3, 16, 17], [4, 5, 19, 22], [6, 7, 8, 11], [9, 10, 12, 21], [1, 13, 14, 15, 18]]
    alphas = np.logspace(1, 5, 25)
    loglik = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        loglik[i] = kfold_validation(ibds_by_chr, gaps, nSamples, timeBoundDict, alpha, k_fold, Ninit, Tmax, \
               minL_calc, maxL_calc, minL_infer, maxL_infer, step, beta)
        print(f'alpha={alpha}, loglik={loglik[i]}')

    # make a plot for easy visulization of how the k-fold likelihood changes as a function of alpha
    plt.plot(alphas, loglik, marker='o')
    plt.xlabel('alpha')
    plt.ylabel('loglikelihood')
    plt.xscale('log')
    plt.savefig(os.path.join(outfolder, 'hyperparam.pdf'), dpi=300)

    return alphas[np.argmax(loglik)]

def hyperparam_opt2(ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict, Ninit=500, Tmax=100, \
                    minL_calc=2.0, maxL_calc=22, minL_infer=6.0, maxL_infer=20.0,
                    FP=None, R=None, POWER=None, step=0.1, beta=250, prefix="", outfolder="", history=False):
    alphas = np.logspace(np.log10(50), 5, 25)
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
        print(f's={s}, e={e}')
        # since here we only want the likelihood, not the penalty term, we can safely set Nconst, alpha, beta all to 0
        loglike = singlePop_MultiTP_given_vecNe_negLoglik_noPenalty(Ne, histograms, binMidpoint, G, gaps, nSamples, Tmax, timeBoundDict,\
                                                                    s=s, e=e, FP=FP, R=R, POWER=POWER)
        loglik[i] = -loglike
        #print(f'{alpha}: {-loglike}')
        Nes.append(Ne)
    plt.plot(alphas, loglik, marker='o')
    plt.xlabel('alpha')
    plt.ylabel('loglikelihood')
    plt.xscale('log')

    delimiter = '.'
    plt.savefig(os.path.join(outfolder, delimiter.join([s for s in [prefix, "hyperparam2_normalized.pdf"] if s.strip()])), dpi=300)
    # save the numpy array loglik to a file
    np.save(os.path.join(outfolder, delimiter.join([s for s in [prefix, 'hyperparam2_normalized.npy'] if s.strip()])), loglik)
    if history:
        pickle.dump(Nes, open(os.path.join(outfolder, \
                        delimiter.join([s for s in [prefix, "hyperparam2_normalized.Nes.pickle"] if s.strip()])), 'wb'))
    
    ########### now, determine the best hyperparam ################
    # first, find the largest alpha that gives a loglikelihood within 1% of the maximum
    max_loglik = np.max(loglik)
    max_alpha_index = np.argmax(loglik)
    print(f'max_loglik={max_loglik}, max_alpha_index={max_alpha_index}, alpha_with_ML={alphas[max_alpha_index]}')
    if alphas[max_alpha_index] < 2500:
        # find the largest alpha that within 0.025 from max_loglik
        opt_alpha_index = np.max(np.where(loglik > max_loglik - 0.02)[0])
        opt_alpha = alphas[opt_alpha_index]
        print(f'route 1: opt_alpha={opt_alpha}')
    else:
        # the highest likelihood is not achieved at the smaller alpha
        opt_alpha = alphas[max_alpha_index]
        print(f'route 2: opt_alpha={opt_alpha}')

    return opt_alpha

# def test_gradient(ibds_by_chr, gaps, nSamples):
#     Ne = np.random.randint(1e3, 1e6, size=100).astype('float64')
#     histograms, binMidpoint, chrlens = prepare_input(ibds_by_chr, np.arange(1,23), ch_len_dict, nSamples.keys(), 4.0, 20.0, 0.01)
#     _, grad = singlePop_MultiTP_given_vecNe_negLoglik(Ne, histograms, binMidpoint, chrlens, gaps, nSamples, 0, 5e4, 1, 1e-4)
    
#     grad_numerical = np.zeros_like(grad)
#     step = 1e-6
#     for i in range(100):
#         Ne1 = np.copy(Ne)
#         Ne2 = np.copy(Ne)
#         Ne1[i] -= step
#         Ne2[i] += step
#         val1, _ = singlePop_MultiTP_given_vecNe_negLoglik(Ne1, histograms, binMidpoint, chrlens, gaps, nSamples, 0, 5e4, 1, 1e-4)
#         val2, _ = singlePop_MultiTP_given_vecNe_negLoglik(Ne2, histograms, binMidpoint, chrlens, gaps, nSamples, 0, 5e4, 1, 1e-4)
#         grad_numerical[i] = (val2 - val1)/(2*step)
#     print(grad - grad_numerical)


################################################### code for estimating cross coal rate #########################################################
#################################################################################################################################################
#################################################################################################################################################

def twoPopIM_MultiTP_given_coalrate_negLoglik(coalrate, histograms, binMidpoint, G, npairs, timeDiff):
    """
    Given a constant coalrate, compute its loglikelihood
    """

    accu = twoPopIM_2tp_given_coalrate_negLoglik(coalrate, histograms, binMidpoint, \
                G, timeDiff, 4*npairs)
    return accu

def inferConstCoalRate_twoPopIM_MultiTP(histograms, binMidpoint, npairs, timeDiff, coalRateInit, chrlens):
    # given the observed ibd segments, estimate the Ne assuming a constant effective Ne
    # gaps: dictionary, where the key is the sampling cluter index and the value is the sampling time
    # nSamples: dictionary, where the key is the sampling cluster index and the value is the number of diploid samples within each cluster
    kargs = (histograms, binMidpoint, chrlens, npairs, timeDiff)
    res = minimize(twoPopIM_MultiTP_given_coalrate_negLoglik, coalRateInit, args=kargs, method='L-BFGS-B', bounds=[(1e-10, 0.1)])
    return res.x[0]

def twoPop_IM_given_vecCoalRates_negLoglik(coalRates, histograms, binMidpoint, G, npairs, time1, time2,\
        alpha, beta, timeBound=None, s=0, e=-1, FP=None, R=None, POWER=None):
    # return the negative loglikelihood of Ne, plus the penalty term
    # G: a vector of chromosome length
    # gaps: dictionary, where the key is the sampling cluter index and the value is the sampling time
    # nSamples: dictionary, where the key is the sampling cluster index and the value is the number of diploid samples within each cluster

    accu, grad = twoPopIM_given_vecCoalRates_negLoglik_noPenalty(coalRates, \
                histograms, binMidpoint, G, 4*npairs,\
                time1, time2, timeBound, s=s, e=e, FP=FP, R=R, POWER=POWER)
    
    coalRates_ = np.copy(coalRates)

    if alpha != 0 :
        penalty1 = alpha*np.sum(np.diff(coalRates_, n=2)**2)
        accu += penalty1

        # add gradient due to the penalty term
        coalRates_left2 = shift(coalRates_, -2, cval=0)
        coalRates_left1 = shift(coalRates_, -1, cval=0)
        coalRates_right2 = shift(coalRates_, 2, cval=0)
        coalRates_right1 = shift(coalRates_, 1, cval=0)
        penalty_grad1 = 12*coalRates_ - 8*(coalRates_left1 + coalRates_right1) + 2*(coalRates_left2 + coalRates_right2)
        penalty_grad1[0] = 2*coalRates_[0]-4*coalRates_[1]+2*coalRates_[2]
        penalty_grad1[1] = 10*coalRates_[1]-4*coalRates_[0]-8*coalRates_[2]+2*coalRates_[3]
        penalty_grad1[-1] = 2*coalRates_[-1]-4*coalRates_[-2]+2*coalRates_[-3]
        penalty_grad1[-2] = 10*coalRates_[-2]-4*coalRates_[-1]-8*coalRates_[-3]+2*coalRates_[-4]
        grad += alpha*penalty_grad1
    if beta != 0:
        penalty2 = beta*np.sum(np.diff(coalRates_, n=1)**2)
        accu += penalty2

        penalty_grad2 = 4*coalRates_ - 2*shift(coalRates_, 1, cval=0) - 2*shift(coalRates_, -1, cval=0)
        penalty_grad2[0] = -(2*coalRates_[1] - 2*coalRates_[0])
        penalty_grad2[-1] = 2*coalRates_[-1] - 2*coalRates_[-2]
        grad += beta*penalty_grad2
        
    return accu, grad

def prepare_input_twoPopIM(ibds_by_chr, ch_ids, ch_len_dict, minL=4.0, maxL=20.0, step=0.1):
    ibds = []
    
    for ch in ch_ids:
        if ibds_by_chr.get(ch):
            ibds.extend(ibds_by_chr[ch])

    bins = np.arange(minL, maxL+step, step=step)
    binMidpoint = (bins[1:] + bins[:-1])/2
    histogram, _ = np.histogram(ibds, bins=bins)
    
    chrlens = [ch_len_dict[ch] for ch in ch_ids]
    return histogram, binMidpoint, np.array(chrlens)



def bootstrap_single_run_twoPopIM(ch_ids, ibds_by_chr, npairs, time1, time2, ch_len_dict, timeBound=None, coalRateInit=1e-3, Tmax=100, \
        minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, \
        step=0.1, alpha=1e10, beta=1e10, FP=None, R=None, POWER=None, verbose=False):
    # perform one bootstrap resampling
    # check that time has been "normalized"
    assert(min(time1, time2) == 0)

    ###################### First, infer a const Ne ################################
    histograms, binMidpoint, chrlens = prepare_input_twoPopIM(ibds_by_chr, ch_ids, ch_len_dict, minL=minL_infer, maxL=maxL_infer, step=step)
    coalRateConst = inferConstCoalRate_twoPopIM_MultiTP(histograms, binMidpoint, npairs, abs(time1 - time2), coalRateInit, chrlens)
    if verbose:
        print(f'estimated constant coalescent rate: {coalRateConst}')
    ################################################################################
    histograms, binMidpoint, chrlens = prepare_input_twoPopIM(ibds_by_chr, ch_ids, ch_len_dict, minL=minL_calc, maxL=maxL_calc, step=step)

    bins_infer = np.arange(minL_infer, maxL_infer+step, step)
    binMidpoint_infer = (bins_infer[1:] + bins_infer[:-1])/2
    s = np.where(np.isclose(binMidpoint, binMidpoint_infer[0]))[0][0]
    e = np.where(np.isclose(binMidpoint, binMidpoint_infer[-1]))[0][0]
    if (not FP is None) and (not R is None) and (not POWER is None):
        assert(len(FP) == len(binMidpoint))
        assert(len(POWER) == len(binMidpoint))
        assert(R.shape[0] == R.shape[1] == len(binMidpoint))

    if timeBound != None:
        low1, high1 = time1 + timeBound[0][0], time1 + timeBound[0][1]
        low2, high2 = time2 + timeBound[1][0], time2 + timeBound[1][1]
        startingTime = max(low1, low2) # this is the most recent time point for which pop1 and pop2 are temporally overlapping, so this is the time point from which cross-coalescence rate will be estimated backward in time
        vecl = Tmax + (max(high1, high2) - startingTime)
        time1 -= startingTime
        time2 -= startingTime # normalize with respect to starting time. this should make it easier to index into coalRates vector.
        #### print some sanity check info ####
        print(f'starting time: {startingTime}')
        print(f'after normalizing, time1: {time1}, time2: {time2}')
        print(f'vector length: {vecl}')

    kargs = (histograms, binMidpoint, chrlens, npairs, time1, time2, alpha, beta, timeBound, s, e+1, FP, R, POWER)
    Nconst = 1/(2*coalRateConst)
    coalInitVec = 1/(2*np.exp(np.random.normal(np.log(Nconst), np.log(Nconst)/25, vecl)))
    res = minimize(twoPop_IM_given_vecCoalRates_negLoglik, coalInitVec, \
        args=kargs, method='L-BFGS-B', jac=True, bounds=[(1e-10, 0.1) for i in range(vecl)], \
        options={'maxfun': 50000, 'maxiter':50000})
    if verbose:
        print(res)
    return res.x




def inferCoalRates_twoPopIM_twoTP(ibds_by_chr, npairs, time1, time2, ch_len_dict, timeBound=None, coalRateInit=1e-3, Tmax=100, \
        minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, step=0.1, alpha=1e10, beta=1e10, FP=None, R=None, POWER=None, \
        doBootstrap=True, nprocess=6, plot=True, outFolder=".", prefix=""):

    """
    This function is to infer cross-coalescence rate between two populations.

    Parameters
    ----------
    ibds_by_chr: dict
        This dictionary should contain a key-value pair for each of the autosomes (the key should be the same as in ch_len_dict). And for each autosome, ibds_by_chr[ch] contain a list of segment lengths. It should only include segments that are shared across the two populations of interest. Not segments that are shared within either of the two populations.
    npairs: int
        Number of (diploid) sample pairs, e.g, for IBD segments found within a sample of size N, this should be N(N-1)/2
    """

    if ((FP is None) or (R is None) or (POWER is None)) and (minL_calc != minL_infer or maxL_calc != maxL_infer):
        warnings.warn('Error model not provided... Setting the length range used for calculation and inference to be the same.')
        minL_calc = minL_infer
        maxL_calc = maxL_infer

    # estimate Ne using the original data
    ch_ids = [k for k, v in ch_len_dict.items()]
    coals = bootstrap_single_run_twoPopIM(ch_ids, ibds_by_chr, npairs, time1, time2, ch_len_dict, timeBound=timeBound, \
        coalRateInit=coalRateInit, Tmax=Tmax, minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, \
        step=step, alpha=alpha, beta=beta, FP=FP, R=R, POWER=POWER, verbose=True)

    if plot:
        plotPosteriorTMRCA(coals, abs(time1 - time2), minL=minL_infer, maxL=maxL_infer, step=step, outFolder=outFolder, prefix=prefix)
        plot2PopIMfit(coals, ibds_by_chr, ch_len_dict, npairs, time1, time2, timeBound=timeBound, minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, FP=FP, R=R, POWER=POWER, outFolder=outFolder, prefix=prefix)

    # start bootstrapping
    if doBootstrap:
        nresample = 200
        ch_ids = [k for k, v in ch_len_dict.items()]
        resample_chrs = [ np.random.choice(ch_ids, size=len(ch_ids)) for i in range(nresample)]
        lowCoals, upCoals, bootstrapFullResult = bootstrap(bootstrap_single_run_twoPopIM, resample_chrs, nprocess, True, ibds_by_chr, npairs, \
                time1, time2, ch_len_dict, timeBound, coalRateInit, Tmax, \
                minL_calc, maxL_calc, minL_infer, maxL_infer, step, alpha, beta, FP, R, POWER)
        return coals, lowCoals, upCoals, bootstrapFullResult
    else:
        return coals

def inferCoalRates_twoPopIM(ibds, nSamples1, nSamples2, time1, time2, ch_len_dict, timeBound=None, coalRateInit=1e-3, Tmax=100, \
        minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, step=0.1, alpha=5e10, beta=1e10, FP=None, R=None, POWER=None, \
        doBootstrap=True, nprocess=6, plot=True, outFolder=".", prefix=""):
    """
    For two populations, infer both within- and cross-population coalescent rate.

    Parameters
    ----------
    ibds: dict
        This should be a dictionary of dictionary. For the first level, it should have exactly three keys, (0,0), (1,1) and (0,1). Each represents within population 1, within population 2, and across population 1 and 2. For the value mapped to each of the three keys, it shuold be a dictionary of the corresponding IBD segments, stored separately for each autosomes. For example, ibds[(0,0)][20] should contain a list of segment lengths of IBD segments found on chr20 within population 1.
    nSamples1: int
        Number of samples in population 1
    nSamples2: int
        Number of samples in population 2
    time1: int
        Sampling time of population 1, expressed in terms of generations before present
    time2: int
        Sampling time of population2, expressed in terms of generations before present
    ch_len_dict: dict
        A dictionary of chromosome length (in cM). The key-value pair is the genetic length (value) of each chromosome (key).
    timeBound: list of two tuples
        Specifies the sampling time uncertainty of the two populations. For example, [(-2,2), (-3,3)] states that the first population's
        dating interval is -2, 2 generations away from $time1, and similarly the second population's dating interval is -3, 3 generations away from $time2.
    coalRateInit: float
        Initial search value for coalescence rates.
    Tmax: int
        Maximum number of generations backward in time for which to infer coalescence rates.
    minL_calc: float
    maxL_calc: float
    minL_infer: float
    maxL_infer: float
    step: float
    alpha: float
    beta: float
    FP: np.array
    R: np.array
    POWER: np.array
    doBootstrap: bool
    nprocess: int
        Number of processes to use
    plot: bool
    outFolder: str
    prefix: str


    """
    if timeBound == None:
        timeBound = [(0,0), (0,0)]

    #### infer coalescent rate within population 1
    if len(prefix) == 0:
        prefix1 = 'pop1'
    else:
        prefix1 = prefix + '.pop1'
    ret = inferCoalRates_twoPopIM_twoTP(ibds[(0,0)], nSamples1*(nSamples1-1)/2, 0, 0, ch_len_dict, \
        timeBound=[timeBound[0], timeBound[0]],\
        coalRateInit=coalRateInit, Tmax=Tmax, minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, \
        step=step, alpha=alpha, beta=beta, FP=FP, R=R, POWER=POWER, \
        doBootstrap=doBootstrap, nprocess=nprocess, plot=plot, outFolder=outFolder, prefix=prefix1)
    if doBootstrap:
        coals1_, lows1_, highs1_, fullBoot1 = ret
    else:
        coals1_ = ret

    #### infer coalescent rate within population 2
    if len(prefix) == 0:
        prefix2 = 'pop2'
    else:
        prefix2 = prefix + '.pop2'
    ret = inferCoalRates_twoPopIM_twoTP(ibds[(1,1)], nSamples2*(nSamples2-1)/2, 0, 0, ch_len_dict, \
        timeBound=[timeBound[1], timeBound[1]], \
        coalRateInit=coalRateInit, Tmax=Tmax, minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, \
        step=step, alpha=alpha, beta=beta, FP=FP, R=R, POWER=POWER, \
        doBootstrap=doBootstrap, nprocess=nprocess, plot=plot, outFolder=outFolder, prefix=prefix2)
    if doBootstrap:
        coals2_, lows2_, highs2_, fullBoot2 = ret
    else:
        coals2_ = ret

    #### infer coalescent rate across pop1 and pop2
    # normalize time
    time1 -= min(time1, time2)
    time2 -= min(time1, time2)

    if len(prefix) == 0:
        prefix12 = 'pop12'
    else:
        prefix12 = prefix + '.pop12'
    ret = inferCoalRates_twoPopIM_twoTP(ibds[(0,1)], nSamples1*nSamples2, time1, time2, ch_len_dict, timeBound=timeBound, \
        coalRateInit=coalRateInit, Tmax=Tmax, minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, \
        step=step, alpha=alpha, beta=beta, FP=FP, R=R, POWER=POWER, \
        doBootstrap=doBootstrap, nprocess=nprocess, plot=plot, outFolder=outFolder, prefix=prefix12)
    if doBootstrap:
        coals12_, lows12_, highs12_, fullBoot12 = ret
    else:
        coals12_ = ret

    #### gather some summary of the output

    offset = min(abs(timeBound[0][0]), abs(timeBound[1][0]))
    time1 += offset
    time2 += offset
    vecl = Tmax + max(time1 + timeBound[0][1], time2 + timeBound[1][1])

    if len(prefix) == 0:
        fname = f'{outFolder}/twoPopIM.coalescence_rates.txt'
    else:
        fname = f'{outFolder}/{prefix}.twoPopIM.coalescence_rates.txt'

    if doBootstrap:
        coals1, lows1, highs1 = np.full(vecl, np.nan), np.full(vecl, np.nan), np.full(vecl, np.nan)
        start = time1 + timeBound[0][0]
        end = time1 + timeBound[0][1] + Tmax
        coals1[start:end] = coals1_
        lows1[start:end] = lows1_
        highs1[start:end] = highs1_

        coals2, lows2, highs2 = np.full(vecl, np.nan), np.full(vecl, np.nan), np.full(vecl, np.nan)
        start = time2 + timeBound[1][0]
        end = time2 + timeBound[1][1] + Tmax
        coals2[start:end] = coals2_
        lows2[start:end] = lows2_
        highs2[start:end] = highs2_

        coals12, lows12, highs12 = np.full(vecl, np.nan), np.full(vecl, np.nan), np.full(vecl, np.nan)
        s = max(time1+timeBound[0][0], time2+timeBound[1][0])
        coals12[s:s+len(coals12_)] = coals12_
        lows12[s:s+len(lows12_)] = lows12_
        highs12[s:s+len(highs12_)] = highs12_

        ###### now compute R and its CI, where R is defined by 2*lambda_12/(lambda_11+lambda_22)
        # diff = abs(time1 - time2)
        # if diff > 0:
        #     if time1 > time2:
        #         Rboot = 2*fullBoot12[:, :-diff]/(fullBoot1[:,:-diff] + fullBoot2[:,diff:])
        #         R_ = 2*coals12_[:-diff]/(coals1_[:-diff] + coals2_[diff:])
        #     else:
        #         Rboot = 2*fullBoot12[:, :-diff]/(fullBoot1[:, diff:] + fullBoot2[:, :-diff])
        #         R_ = 2*coals12_[:-diff]/(coals1_[diff:] + coals2_[:-diff])
        # else:
        #     Rboot = 2*fullBoot12/(fullBoot1 + fullBoot2)
        #     R_ = 2*coals12_/(coals1_ + coals2_)

        # nSamples = Rboot.shape[0]
        # Rsorted = np.sort(Rboot, axis=0)
        # index = int(2.5/(100/nSamples))
        # R, lowsR, highsR = np.full(vecl, np.nan), np.full(vecl, np.nan), np.full(vecl, np.nan)
        # R[s:s+Tmax-diff] = R_
        # lowsR[s:s+Tmax-diff] = Rsorted[index-1]
        # highsR[s:s+Tmax-diff] = Rsorted[-index]

        ret = pd.DataFrame(pd.DataFrame({'Generation': np.arange(1, 1 + vecl), 
                    'pop1_coalrate': coals1, 'pop1_coalrate_lowCI': lows1, 'pop1_coalrate_highCI': highs1, 
                    'pop2_coalrate': coals2, 'pop2_coalrate_lowCI': lows2, 'pop2_coalrate_highCI': highs2, 
                    'cross_coalrate': coals12, 'cross_coalrate_lowCI': lows12, 'cross_coalrate_highCI': highs12}))
        ret.to_csv(fname, index=False)
    else:
        coals1, coals2, coals12 = np.full(vecl, np.nan), np.full(vecl, np.nan), np.full(vecl, np.nan)
        coals1[time1 + timeBound[0][0]:time1 + timeBound[0][1] + Tmax] = coals1_
        coals2[time2 + timeBound[1][0]:time2 + timeBound[1][1] + Tmax] = coals2_
        s = max(time1+timeBound[0][0], time2+timeBound[1][0])
        coals12[s:s+len(coals12_)] = coals12_
        
        # diff = abs(time1 - time2)
        # if diff > 0:
        #     if time1 > time2:
        #         R_ = 2*coals12_[:-diff]/(coals1_[:-diff] + coals2_[diff:])
        #     else:
        #         R_ = 2*coals12_[:-diff]/(coals1_[diff:] + coals2_[:-diff])
        # else:
        #     R_ = 2*coals12_/(coals1_ + coals2_)
        # R[s:s+Tmax-diff] = R_
        ret = pd.DataFrame(pd.DataFrame({'Generation': np.arange(1, 1 + vecl), 
                    'pop1_coalrate': coals1, 
                    'pop2_coalrate': coals2, 
                    'cross_coalrate': coals12}))
        ret.to_csv(fname, index=False)


def maskIBD(path2IBD, path2ChrDelimiter, path2mask=None):
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
        ibds_by_chr = defaultdict(list)
        with open(path2IBD) as f:
            for line in f:
                iid1, iid2, ch, ibd_startCM, ibd_endCM, lengthCM = line.strip().split()
                ch, ibd_startCM, ibd_endCM, lengthCM = int(ch), float(ibd_startCM), float(ibd_endCM), float(lengthCM)
                for i, span in enumerate(spanlist[map[ch]:map[ch+1]]):
                    # ask whether the current segment being read has any overlap with this span of interest
                    span_start, span_end = span
                    if ibd_endCM <= span_start or ibd_startCM >= span_end:
                        continue
                    else:
                        ibd_start_masked, ibd_end_masked = max(span_start, ibd_startCM), min(span_end, ibd_endCM)
                        ibds_by_chr[i+map[ch]].append(ibd_end_masked - ibd_start_masked)
    
    else:
        ########## no mask file provided, thus no masking will be performed ###################
        print(f'no mask file provided, all segments will be used for inference')
        ch_len_dict = {k:v[1]-v[0] for k, v in chrDelimiter.items()}
        ibds_by_chr = defaultdict(dict)
        with open(path2IBD) as f:
            for line in f:
                iid1, iid2, ch, _, _, lengthCM = line.strip().split()
                ch, lengthCM = int(ch), float(lengthCM)
                ibds_by_chr[ch].append(lengthCM)
               
    return ch_len_dict, ibds_by_chr

def inferCoalRate_twoPopIM_withMask(ibds_pop1, ibds_pop2, ibds_pop12, nSamples1, nSamples2, time1, time2, path2ChrDelimiter, path2mask=None, \
        coalRateInit=1e-3, Tmax=100, minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, \
        step=0.1, alpha=5e10, beta=1e10, FP=None, R=None, POWER=None, \
        doBootstrap=True, nprocess=6, plot=True, outFolder=".", prefix=""):
    ch_len_dict, ibds_by_chr_pop1 = maskIBD(ibds_pop1, path2ChrDelimiter, path2mask=path2mask)
    _, ibds_by_chr_pop2 = maskIBD(ibds_pop2, path2ChrDelimiter, path2mask=path2mask)
    _, ibds_by_chr_pop12 = maskIBD(ibds_pop12, path2ChrDelimiter, path2mask=path2mask)

    ibds = {}
    ibds[(0,0)] = ibds_by_chr_pop1
    ibds[(1,1)] = ibds_by_chr_pop2
    ibds[(0,1)] = ibds_by_chr_pop12

    inferCoalRates_twoPopIM(ibds, nSamples1, nSamples2, time1, time2, ch_len_dict, coalRateInit=coalRateInit, Tmax=Tmax, \
        minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, alpha=alpha, beta=beta, \
        FP=FP, R=R, POWER=POWER, doBootstrap=doBootstrap, nprocess=nprocess, plot=plot, outFolder=outFolder, prefix=prefix)


