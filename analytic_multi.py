import numpy as np
import itertools
from collections import defaultdict
from scipy.optimize import minimize
from scipy.ndimage import shift
from scipy.stats import norm
from analytic import singlePop_2tp_given_Ne_negLoglik, singlePop_2tp_given_vecNe_negLoglik_noPenalty, twoPopIM_2tp_given_coalrate_negLoglik, twoPopIM_given_vecCoalRates_negLoglik_noPenalty
from ts_utility import multi_run
from plot import plot_pairwise_fitting, plot_pairwise_TMRCA, plotPosteriorTMRCA, plot2PopIMfit
from AccProx import AccProx_trendFiltering_l1
import matplotlib.pyplot as plt
import pickle
import warnings
from pathlib import Path
import pandas as pd
import itertools

ch_len_dict = {1:286.279, 2:268.840, 3:223.361, 4:214.688, 5:204.089, 6:192.040, 7:187.221, 8:168.003, 9:166.359, \
        10:181.144, 11:158.219, 12:174.679, 13:125.706, 14:120.203, 15:141.860, 16:134.038, 17:128.491, 18:117.709, \
        19:107.734, 20:108.267, 21:62.786, 22:74.110}


def singlePop_MultiTP_given_Ne_negLoglik(Ne, histograms, binMidpoint, G, gaps, nSamples, timeBoundDict):
    accu = 0
    for id, nSample in nSamples.items():
        if nSample == 1:
            continue
        accu += singlePop_2tp_given_Ne_negLoglik(Ne, histograms[(id, id)], binMidpoint, G, gaps[id], gaps[id],\
            (2*nSample)*(2*nSample-2)/2, [(timeBoundDict[id]), (timeBoundDict[id])])
    for id1, id2 in itertools.combinations(nSamples.keys(), 2):
        accu += singlePop_2tp_given_Ne_negLoglik(Ne, histograms[(min(id1, id2), max(id1, id2))], binMidpoint, \
                G, gaps[id1], gaps[id2], (2*nSamples[id1])*(2*nSamples[id2]),\
                [(timeBoundDict[id1]), (timeBoundDict[id2])])
    return accu

def inferConstNe_singlePop_MultiTP(histograms, binMidpoint, gaps, nSamples, Ninit, chrlens, timeBoundDict):
    # given the observed ibd segments, estimate the Ne assuming a constant effective Ne
    # gaps: dictionary, where the key is the sampling cluter index and the value is the sampling time
    # nSamples: dictionary, where the key is the sampling cluster index and the value is the number of diploid samples within each cluster

    kargs = (histograms, binMidpoint, chrlens, gaps, nSamples, timeBoundDict)
    res = minimize(singlePop_MultiTP_given_Ne_negLoglik, Ninit, args=kargs, method='L-BFGS-B', bounds=[(10, 5e6)])
    return res.x[0]


def singlePop_MultiTP_given_vecNe_negLoglik(Ne, histograms, binMidpoint, G, gaps, nSamples, Tmax, alpha, beta, timeBoundDict, s=0, e=-1, FP=None, R=None, POWER=None, tail=False):
    # return the negative loglikelihood of Ne, plus the penalty term
    # G: a vector of chromosome length
    # gaps: dictionary, where the key is the sampling cluter index and the value is the sampling time
    # nSamples: dictionary, where the key is the sampling cluster index and the value is the number of diploid samples within each cluster
    accu = 0
    grad = np.zeros_like(Ne)
    for id, nSample in nSamples.items():
        if nSample == 1:
            continue
        age = gaps[id]
        accu_, grad_ = singlePop_2tp_given_vecNe_negLoglik_noPenalty(Ne, histograms[(id, id)], binMidpoint, G, \
            age, age, Tmax, (2*nSample)*(2*nSample-2)/2, [timeBoundDict[id], timeBoundDict[id]], 
            s=s, e=e, FP=FP, R=R, POWER=POWER, tail=tail)
        accu += accu_
        grad += grad_
    for id1, id2 in itertools.combinations(nSamples.keys(), 2):
        #i = max(gaps[id1], gaps[id2])-time_offset
        accu_, grad_ = singlePop_2tp_given_vecNe_negLoglik_noPenalty(Ne, histograms[(min(id1, id2), max(id1, id2))], \
            binMidpoint, G, gaps[id1], gaps[id2], Tmax, \
                (2*nSamples[id1])*(2*nSamples[id2]), [timeBoundDict[id1], timeBoundDict[id2]], 
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

    return accu, grad

def bootstrap_single_run(ch_ids, ibds_by_chr, gaps, nSamples, ch_len_dict, timeBoundDict, Ninit=500, Tmax=100, \
        minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, \
        step=0.1, alpha=1e-6, beta=1e-4, method='l2', FP=None, R=None, POWER=None, verbose=False):
    # perform one bootstrap resampling

    ###################### First, infer a const Ne ################################
    histograms, binMidpoint, chrlens = prepare_input(ibds_by_chr, ch_ids, ch_len_dict, nSamples.keys(), minL=minL_infer, maxL=maxL_infer, step=step)
    Nconst = inferConstNe_singlePop_MultiTP(histograms, binMidpoint, gaps, nSamples, Ninit, chrlens, timeBoundDict)
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


    bins_infer = np.arange(minL_infer, maxL_infer+step, step)
    binMidpoint_infer = (bins_infer[1:] + bins_infer[:-1])/2
    s = np.where(np.isclose(binMidpoint, binMidpoint_infer[0]))[0][0]
    e = np.where(np.isclose(binMidpoint, binMidpoint_infer[-1]))[0][0]
    if (not FP is None) and (not R is None) and (not POWER is None):
        assert(len(FP) == len(binMidpoint))
        assert(len(POWER) == len(binMidpoint))
        assert(R.shape[0] == R.shape[1] == len(binMidpoint))

    kargs = (histograms, binMidpoint, chrlens, gaps_adjusted, nSamples, Tmax, alpha, beta, timeBoundDict, s, e+1, FP, R, POWER)
    NinitVec = np.exp(np.random.normal(np.log(Nconst), np.log(Nconst)/25, Tmax + (t_max - t_min)))
    
    if method == 'l2':
        res = minimize(singlePop_MultiTP_given_vecNe_negLoglik, NinitVec, \
                args=kargs, method='L-BFGS-B', jac=True, bounds=[(10, 5e6) for i in range(Tmax + (t_max - t_min))], options={'maxfun': 50000, 'maxiter':50000})
        if verbose:
            print(res)
        return res.x
    elif method == 'l1':
        res = AccProx_trendFiltering_l1(NinitVec, *kargs)
        if verbose:
            print(res)
        return res
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


def inferVecNe_singlePop_MultiTP(ibds_by_chr, gaps, nSamples, ch_len_dict, timeBound=None, Ninit=500, Tmax=100, \
        minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0,
        step=0.1, alpha=1e-6, beta=1e-4, method='l2', FP=None, R=None, POWER=None, nprocess=6, plot=False, prefix="", doBootstrap=True, outFolder='.'):
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
    if not timeBound:
        timeBound = {}
        for id, _ in nSamples.items():
            timeBound[id] = (0, 0)

    assert(np.min(list(gaps.values())) == 0) # the most recent sample cluster must be labeled as generation 0
    Ne = bootstrap_single_run(ch_ids, ibds_by_chr, gaps, nSamples, ch_len_dict, timeBound, Ninit=Ninit, Tmax=Tmax, \
        minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, \
        step=step, alpha=alpha, beta=beta, method=method, FP=FP, R=R, POWER=POWER, verbose=True)
    if plot:
        if len(prefix) > 0:
            plot_prefix = "pairwise." + prefix
        else:
            plot_prefix = "pairwise"
        plot_pairwise_fitting(ibds_by_chr, gaps, nSamples, ch_len_dict, Ne, outFolder, prefix=plot_prefix, \
            minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL=maxL_infer, step=step, minL_plot=6.0, FP=FP, R=R, POWER=POWER)
        plot_pairwise_TMRCA(gaps, Ne, Tmax, outFolder, prefix=plot_prefix, minL=minL_infer, maxL=maxL_infer, step=0.25)

    if doBootstrap:
        # start bootstrapping
        nresample = 200
        ch_ids = [k for k, v in ch_len_dict.items()]
        resample_chrs = [ np.random.choice(ch_ids, size=len(ch_ids)) for i in range(nresample)]
        lowCI, upCI = bootstrap(bootstrap_single_run, resample_chrs, nprocess, False, ibds_by_chr, gaps, nSamples, ch_len_dict, timeBound, Ninit, Tmax, \
            minL_calc, maxL_calc, minL_infer, maxL_infer, step, alpha, beta, method, FP, R, POWER)
        return Ne, lowCI, upCI
    else:
        return Ne


def inferVecNe_singlePop_MultiTP_withMask(path2IBD, path2ChrDelimiter, path2mask=None, nSamples=-1, path2SampleAge=None, \
        Ninit=500, Tmax=100, minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, step=0.1, alpha=2500, beta=0, method='l2', \
        FP=None, R=None, POWER=None, generation_time=29, minSample=10, maxSample=np.inf, merge_level=5, \
        prefix="", doBootstrap=True, outFolder='.'):
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
            print(f'{nSamples[i]} samples in age bin {i}: {[k for k, v in sampleAgeBinDict.items() if v == i]}')
        
        for i in exSampleCluster:
            del nSamples[i]
            del gaps[i]
        print(f'Bins {[i for i, _ in gaps.items()]} will be used for inference.')
        
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


    ### end of preprocessing of input data, ready to start the inference
    if doBootstrap:
        Ne, lowCI, highCI = inferVecNe_singlePop_MultiTP(ibds_by_chr, gaps, nSamples, ch_len_dict, Ninit=Ninit, Tmax=Tmax, \
                minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, alpha=alpha, beta=beta, \
                method=method, FP=FP, R=R, POWER=POWER, plot=True, prefix=prefix, doBootstrap=doBootstrap, outFolder=outFolder)
        return Ne, lowCI, highCI
    else:
        Ne = inferVecNe_singlePop_MultiTP(ibds_by_chr, gaps, nSamples, ch_len_dict, Ninit=Ninit, Tmax=Tmax, \
                minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, alpha=alpha, beta=beta, \
                method=method, FP=FP, R=R, POWER=POWER, plot=True, prefix=prefix, doBootstrap=doBootstrap, outFolder=outFolder)
        return Ne



# def kfold_validation(ibds_by_chr, gaps, nSamples, alpha, k_fold, Ninit=500, Tmax=100, minL=4.0, maxL=20.0, step=0.1, beta=1e-4):
#     time_offset = np.min(list(gaps.values()))
#     # return the likelihood averaged over the 5 validation sets with the given alpha
#     results = np.zeros(len(k_fold))
#     for i, validation_set in enumerate(k_fold):
#         train_set = [i for i in np.arange(1, 23) if i not in validation_set]
#         Ne = bootstrap_single_run(ibds_by_chr, train_set, gaps, nSamples, Ninit, Tmax, minL, maxL, step, alpha, beta)
#         histograms, binMidpoint, G = prepare_input(ibds_by_chr, validation_set, ch_len_dict, nSamples.keys(), minL, maxL, step)
#         # since here we only want the likelihood, not the penalty term, we can safely set Nconst, alpha, beta all to 0
#         loglike, _ = singlePop_MultiTP_given_vecNe_negLoglik(Ne, histograms, binMidpoint, G, gaps, nSamples, time_offset, 0, 0, 0)
#         results[i] = -loglike # the function returns the negative of loglikelihood, so we need to reverse the sign
#     print(f'{alpha}: {np.mean(results)}')
#     return np.mean(results) 


# def hyperparam_opt(ibds_by_chr, gaps, nSamples, Ninit=500, Tmax=100, minL=4.0, maxL=20.0, step=0.1, beta=1e-4, nprocess=6, outfolder="", prefix=""):
#     k_fold = [[2, 3, 16, 17], [4, 5, 19, 22], [6, 7, 8, 11], [9, 10, 12, 21], [1, 13, 14, 15, 18]]
#     alphas = np.logspace(0, 3, 100)
#     params = [[ibds_by_chr, gaps, nSamples, alpha, k_fold, Ninit, Tmax, minL, maxL, step, beta] for alpha in alphas]
#     results = multi_run(kfold_validation, params, processes=nprocess)

#     # make a plot for easy visulization of how the k-fold likelihood changes as a function of alpha
#     plt.plot(alphas, results, marker='o')
#     plt.xlabel('alpha')
#     plt.ylabel('loglikelihood')
#     plt.xscale('log')
#     plt.savefig(f'{outfolder}/hyperparam.{prefix}.pdf', dpi=300)

#     return alphas[np.argmax(results)]


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

def twoPop_IM_given_vecCoalRates_negLoglik(coalRates, histograms, binMidpoint, G, npairs, timeGap,\
        alpha, beta, s=0, e=-1, FP=None, R=None, POWER=None):
    # return the negative loglikelihood of Ne, plus the penalty term
    # G: a vector of chromosome length
    # gaps: dictionary, where the key is the sampling cluter index and the value is the sampling time
    # nSamples: dictionary, where the key is the sampling cluster index and the value is the number of diploid samples within each cluster

    accu, grad = twoPopIM_given_vecCoalRates_negLoglik_noPenalty(coalRates, \
                histograms, binMidpoint, G, 4*npairs,\
                timeGap, s=s, e=e, FP=FP, R=R, POWER=POWER)
    
    ### try log transform
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



def bootstrap_single_run_twoPopIM(ch_ids, ibds_by_chr, npairs, timeDiff, ch_len_dict, coalRateInit=1e-3, Tmax=100, \
        minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, \
        step=0.1, alpha=1e10, beta=1e10, FP=None, R=None, POWER=None, verbose=False):
    # perform one bootstrap resampling

    ###################### First, infer a const Ne ################################
    histograms, binMidpoint, chrlens = prepare_input_twoPopIM(ibds_by_chr, ch_ids, ch_len_dict, minL=minL_infer, maxL=maxL_infer, step=step)
    coalRateConst = inferConstCoalRate_twoPopIM_MultiTP(histograms, binMidpoint, npairs, timeDiff, coalRateInit, chrlens)
    if verbose:
        print(f'estimated constant coalescent rate: {coalRateConst}')
    ################################################################################
    histograms, binMidpoint, chrlens = prepare_input_twoPopIM(ibds_by_chr, ch_ids, ch_len_dict, minL=minL_calc, maxL=maxL_calc, step=step)


    bins_infer = np.arange(minL_infer, maxL_infer+step, step)
    binMidpoint_infer = (bins_infer[1:] + bins_infer[:-1])/2
    s = np.where(np.isclose(binMidpoint, binMidpoint_infer[0]))[0][0]
    e = np.where(np.isclose(binMidpoint, binMidpoint_infer[-1]))[0][0]
    if (not FP is None) and (not R is None) and (not POWER is None):
        print(len(FP))
        print(len(binMidpoint))
        assert(len(FP) == len(binMidpoint))
        assert(len(POWER) == len(binMidpoint))
        assert(R.shape[0] == R.shape[1] == len(binMidpoint))

    kargs = (histograms, binMidpoint, chrlens, npairs, timeDiff, alpha, beta, s, e+1, FP, R, POWER)
    Nconst = 1/(2*coalRateConst)
    coalInitVec = 1/(2*np.exp(np.random.normal(np.log(Nconst), np.log(Nconst)/25, Tmax)))
    res = minimize(twoPop_IM_given_vecCoalRates_negLoglik, coalInitVec, \
        args=kargs, method='L-BFGS-B', jac=True, bounds=[(1e-10, 0.1) for i in range(Tmax)], options={'maxfun': 50000, 'maxiter':50000})
    if verbose:
        print(res)
    return res.x




def inferCoalRates_twoPopIM_twoTP(ibds_by_chr, npairs, timeDiff, ch_len_dict, coalRateInit=1e-3, Tmax=100, \
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
    coals = bootstrap_single_run_twoPopIM(ch_ids, ibds_by_chr, npairs, timeDiff, ch_len_dict, \
        coalRateInit=coalRateInit, Tmax=Tmax, minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, \
        step=step, alpha=alpha, beta=beta, FP=FP, R=R, POWER=POWER, verbose=True)

    if plot:
        plotPosteriorTMRCA(coals, timeDiff, minL=minL_infer, maxL=maxL_infer, step=step, outFolder=outFolder, prefix=prefix)
        plot2PopIMfit(coals, ibds_by_chr, ch_len_dict, timeDiff, npairs, minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, step=step, FP=FP, R=R, POWER=POWER, outFolder=outFolder, prefix=prefix)

    # start bootstrapping
    if doBootstrap:
        nresample = 200
        ch_ids = [k for k, v in ch_len_dict.items()]
        resample_chrs = [ np.random.choice(ch_ids, size=len(ch_ids)) for i in range(nresample)]
        lowCoals, upCoals, bootstrapFullResult = bootstrap(bootstrap_single_run_twoPopIM, resample_chrs, nprocess, True, ibds_by_chr, npairs, \
                timeDiff, ch_len_dict, coalRateInit, Tmax, \
                minL_calc, maxL_calc, minL_infer, maxL_infer, step, alpha, beta, FP, R, POWER)
        return coals, lowCoals, upCoals, bootstrapFullResult
    else:
        return coals

def inferCoalRates_twoPopIM(ibds, nSamples1, nSamples2, time1, time2, ch_len_dict, coalRateInit=1e-3, Tmax=100, \
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

    #### infer coalescent rate within population 1
    if len(prefix) == 0:
        prefix1 = 'pop1'
    else:
        prefix1 = prefix + '.pop1'
    ret = inferCoalRates_twoPopIM_twoTP(ibds[(0,0)], nSamples1*(nSamples1-1)/2, 0, ch_len_dict, \
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
    ret = inferCoalRates_twoPopIM_twoTP(ibds[(1,1)], nSamples2*(nSamples2-1)/2, 0, ch_len_dict, \
        coalRateInit=coalRateInit, Tmax=Tmax, minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, \
        step=step, alpha=alpha, beta=beta, FP=FP, R=R, POWER=POWER, \
        doBootstrap=doBootstrap, nprocess=nprocess, plot=plot, outFolder=outFolder, prefix=prefix2)
    if doBootstrap:
        coals2_, lows2_, highs2_, fullBoot2 = ret
    else:
        coals2_ = ret

    #### infer coalescent rate across pop1 and pop2
    if len(prefix) == 0:
        prefix12 = 'pop12'
    else:
        prefix12 = prefix + '.pop12'
    ret = inferCoalRates_twoPopIM_twoTP(ibds[(0,1)], nSamples1*nSamples2, abs(time1 - time2), ch_len_dict, \
        coalRateInit=coalRateInit, Tmax=Tmax, minL_calc=minL_calc, maxL_calc=maxL_calc, minL_infer=minL_infer, maxL_infer=maxL_infer, \
        step=step, alpha=alpha, beta=beta, FP=FP, R=R, POWER=POWER, \
        doBootstrap=doBootstrap, nprocess=nprocess, plot=plot, outFolder=outFolder, prefix=prefix12)
    if doBootstrap:
        coals12_, lows12_, highs12_, fullBoot12 = ret
    else:
        coals12_ = ret

    #### gather some summary of the output
    if len(prefix) == 0:
        fname = f'{outFolder}/twoPopIM.coalescence_rates.txt'
    else:
        fname = f'{outFolder}/{prefix}.twoPopIM.coalescence_rates.txt'

    # normalize time
    time1 -= min(time1, time2)
    time2 -= min(time1, time2)
    vecl = abs(time1 - time2) + Tmax

    if doBootstrap:
        coals1, lows1, highs1 = np.full(vecl, np.nan), np.full(vecl, np.nan), np.full(vecl, np.nan)
        coals1[time1:time1+Tmax] = coals1_
        lows1[time1:time1+Tmax] = lows1_
        highs1[time1:time1+Tmax] = highs1_

        coals2, lows2, highs2 = np.full(vecl, np.nan), np.full(vecl, np.nan), np.full(vecl, np.nan)
        coals2[time2:time2+Tmax] = coals2_
        lows2[time2:time2+Tmax] = lows2_
        highs2[time2:time2+Tmax] = highs2_

        coals12, lows12, highs12 = np.full(vecl, np.nan), np.full(vecl, np.nan), np.full(vecl, np.nan)
        s = max(time1, time2)
        coals12[s:s+Tmax] = coals12_
        lows12[s:s+Tmax] = lows12_
        highs12[s:s+Tmax] = highs12_

        ###### now compute R and its CI, where R is defined by 2*lambda_12/(lambda_11+lambda_22)
        diff = abs(time1 - time2)
        if diff > 0:
            if time1 > time2:
                Rboot = 2*fullBoot12[:, :-diff]/(fullBoot1[:,:-diff] + fullBoot2[:,diff:])
                R_ = 2*coals12_[:-diff]/(coals1_[:-diff] + coals2_[diff:])
            else:
                Rboot = 2*fullBoot12[:, :-diff]/(fullBoot1[:, diff:] + fullBoot2[:, :-diff])
                R_ = 2*coals12_[:-diff]/(coals1_[diff:] + coals2_[:-diff])
        else:
            Rboot = 2*fullBoot12/(fullBoot1 + fullBoot2)
            R_ = 2*coals12_/(coals1_ + coals2_)

        nSamples = Rboot.shape[0]
        Rsorted = np.sort(Rboot, axis=0)
        index = int(2.5/(100/nSamples))
        R, lowsR, highsR = np.full(vecl, np.nan), np.full(vecl, np.nan), np.full(vecl, np.nan)
        R[s:s+Tmax-diff] = R_
        lowsR[s:s+Tmax-diff] = Rsorted[index-1]
        highsR[s:s+Tmax-diff] = Rsorted[-index]

        ret = pd.DataFrame(pd.DataFrame({'Generation': np.arange(1, 1 + abs(time1-time2) + Tmax), 
                    'pop1_coalrate': coals1, 'pop1_coalrate_lowCI': lows1, 'pop1_coalrate_highCI': highs1, 
                    'pop2_coalrate': coals2, 'pop2_coalrate_lowCI': lows2, 'pop2_coalrate_highCI': highs2, 
                    'cross_coalrate': coals12, 'cross_coalrate_lowCI': lows12, 'cross_coalrate_highCI': highs12,
                    'R': R, 'R_lowCI': lowsR, 'R_highCI': highsR}))
        ret.to_csv(fname, index=False)
    else:
        coals1, coals2, coals12 = np.full(vecl, np.nan), np.full(vecl, np.nan), np.full(vecl, np.nan)
        R = np.full(vecl, np.nan)
        coals1[time1:time1+Tmax] = coals1_
        coals2[time2:time2+Tmax] = coals2_
        s = max(time1, time2)
        coals12[s:s+Tmax] = coals12_
        
        diff = abs(time1 - time2)
        if diff > 0:
            if time1 > time2:
                R_ = 2*coals12_[:-diff]/(coals1_[:-diff] + coals2_[diff:])
            else:
                R_ = 2*coals12_[:-diff]/(coals1_[diff:] + coals2_[:-diff])
        else:
            R_ = 2*coals12_/(coals1_ + coals2_)
        R[s:s+Tmax-diff] = R_
        ret = pd.DataFrame(pd.DataFrame({'Generation': np.arange(1, 1 + abs(time1-time2) + Tmax), 
                    'pop1_coalrate': coals1, 
                    'pop2_coalrate': coals2, 
                    'cross_coalrate': coals12,
                    'R': R}))
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


