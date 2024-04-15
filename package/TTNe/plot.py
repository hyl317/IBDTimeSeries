# some plotting function for ease of visualizing and interpreting the results

import os
import math
import matplotlib.gridspec as gridspec
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from TTNe.analytic import singlePop_2tp_given_vecNe, singlePop_2tp_given_vecNe_withError, computePosteriorTMRCA, twoPopIM_given_coalRate, twoPopIM_given_coalRate_withError
from TTNe.inference_utility import get_ibdHistogram_by_sampleCluster


def plot_pairwise_fitting(df_ibd, sampleCluster2id, dates, nHaplotypePairs, ch_len_dict, estNe, outFolder, timeBoundDict, prefix="pairwise", \
        minL_calc=2.0, maxL_calc=24, minL_infer=6.0, maxL=20.0, step=0.25, minL_plot=6.0, FP=None, R=None, POWER=None):

    bins = np.arange(minL_plot, maxL+step, step)
    midpoint = (bins[1:]+bins[:-1])/2
    histograms = get_ibdHistogram_by_sampleCluster(df_ibd, sampleCluster2id, bins, ch_len_dict)
    L = np.array([v for _, v in ch_len_dict.items()])

    popLabels = dates.keys()
    n = len(popLabels)
    numSubplot = int(n*(n+1)/2)
    # let's say we fix the number of columns to be 5
    ncol = 5
    nrow = math.ceil(numSubplot/5)
    width_ratios = [3] * ncol  # Equal width for each column
    height_ratios = [1] * nrow  # Equal height for each row
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrow, ncol, width_ratios=width_ratios, 
                        height_ratios=height_ratios, wspace=0.15, hspace=0.3)

    # in function inferVecNe_singlePop_MultiTP, we have checked that the min sample time is 0, so this offset is not needed
    # offset = np.inf
    # for k, time in gaps.items():
    #     if time < offset:
    #         offset = time

    ## adjust time if the timebound is specified
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
    dates = dates_adjusted

    # set up the canvas
    for index, pairs in enumerate(itertools.combinations_with_replacement(popLabels, 2)):
        id1, id2 = pairs
        id1, id2 = min(id1, id2), max(id1, id2)
        npairs = nHaplotypePairs[(id1, id2)]/4 # we plot IBD counts for pairs of diploid individuals, so we need to divide by 4
        x = histograms[(id1, id2)]/npairs        
        
        i, j = index//ncol, index%ncol
        ax = fig.add_subplot(gs[i,j])
        
        if index == 0:
            ax.scatter(midpoint, x, label='Observed IBD sharing', s=3.0, color='grey')
        else:
            ax.scatter(midpoint, x, s=3.0, color='grey')

        # plot fit from estimated Ne
        low1, high1 = timeBoundDict[id1]
        low2, high2 = timeBoundDict[id2]
        weight_per_combo = 1/((high1+1-low1)*(high2+1-low2))
        lambda_accu = np.zeros(len(midpoint))
        for i in np.arange(low1, high1+1):
            for j in np.arange(low2, high2+1):
                age_ = max(dates[id1]+i, dates[id2]+j)
                lambda_, _ = singlePop_2tp_given_vecNe(estNe[age_:], L, midpoint, abs(dates[id1]+i - dates[id2]-j))
                lambda_accu += weight_per_combo*lambda_
        meanNumIBD_expectation = 4*(step/100)*lambda_accu
        if index == 0:
            ax.plot(midpoint, meanNumIBD_expectation, label='Expected IBD sharing from inferred Ne', color='red', linewidth=0.75)
        else:
            ax.plot(midpoint, meanNumIBD_expectation, color='red', linewidth=0.75)

        if (not FP is None) and (not R is None) and (not POWER is None):
            bins_calc = np.arange(minL_calc, maxL_calc+step, step)
            binMidpoint = (bins_calc[1:] + bins_calc[:-1])/2
            s = np.where(np.isclose(binMidpoint, midpoint[0]))[0][0]
            e = np.where(np.isclose(binMidpoint, midpoint[-1]))[0][0]
            low1, high1 = timeBoundDict[id1]
            low2, high2 = timeBoundDict[id2]
            weight_per_combo = 1/((high1+1-low1)*(high2+1-low2))
            lambda_accu = np.zeros(len(binMidpoint))
            for i in np.arange(low1, high1+1):
                for j in np.arange(low2, high2+1):
                    age_ = max(dates[id1]+i, dates[id2]+j)
                    lambda_, _ = singlePop_2tp_given_vecNe_withError(estNe[age_:], L, binMidpoint, abs(dates[id1]+i - dates[id2]-j), FP, R, POWER)
                    lambda_accu += weight_per_combo*lambda_
            meanNumIBD_theory = lambda_accu[s:e+1]
            meanNumIBD_theory = 4*(step/100)*meanNumIBD_theory
            if index == 0:
                ax.plot(midpoint, meanNumIBD_theory, color='orange', label='Expected IBD sharing from inferred Ne (with error correction)', linewidth=0.75)
            else:
                ax.plot(midpoint, meanNumIBD_theory, color='orange', linewidth=0.75)


        ##### plot a dashed vertical line to indicate which subset of IBD is used in inference
        ax.axvline(minL_infer, 0, 1, color='k', linestyle='--', linewidth=0.5)

        ax.set_yscale('log')
        title = f'({dates[id1]}, {dates[id2]})' if id1 != id2 else f'{dates[id1]}'
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
    
    plt.figlegend(loc='lower center', fontsize=6)
    plt.subplots_adjust(bottom=0.2)  # Adjust the value as needed
    # add horizontal text at x=0.5, y=0.1
    fig.text(0.5, 0.125, 'IBD Segment Length (cM)', ha='center', va='center', fontsize=12)
    fig.text(0.05, 0.5, '# of Segments per Pair', ha='center', va='center', rotation='vertical', fontsize=12)
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    plt.gcf().set_facecolor('white')
    plt.savefig(os.path.join(outFolder, f'{prefix}.fit.png'), dpi=300, bbox_inches = "tight")
    plt.savefig(os.path.join(outFolder, f'{prefix}.fit.pdf'), bbox_inches = "tight")
    plt.clf()


def find_nearest(arr, val):
    # return index into arr such that the element of that index is numerically closest to val
    residual = np.abs(arr - val)
    return np.argmin(residual)

def closest_indices(a, b):
    c = np.argmin(np.abs(a[:, None] - b), axis=0)
    return c


# def plotPosteriorTMRCA(coalRates, gap=0, minL=6.0, maxL=20.0, step=0.25, outFolder="", prefix=""):
#     """
#     make a heatmap of the posterior TMRCA distribution for segments of length between minL and maxL, given a vector of estimated coalescent rates over generations.
#     """
#     bins = np.arange(minL, maxL+step, step)
#     binMidpoint = (bins[1:] + bins[:-1])/2
#     postprob = computePosteriorTMRCA(coalRates, binMidpoint, gap)

#     outFigPrefix = f"{outFolder}/postTMRCA"
#     if len(prefix) > 0:
#         outFigPrefix = outFigPrefix + "." + prefix
    
#     ##### plot pdf of posterior TMRCA
#     plt.imshow(postprob, cmap='viridis', aspect='auto')
#     plt.colorbar()
#     plt.xlabel('Segment Length (cM)')
#     plt.ylabel('TMRCA (generations backward in time)')
#     bins = np.arange(minL, maxL+step, step)
#     binMidpoint = (bins[1:]+bins[:-1])/2
#     nbins =len(binMidpoint)
#     xticks = np.arange(0, len(binMidpoint), 25)
#     yticks = np.arange(0, len(coalRates), 10)
#     xs = [round(x, 3) for x in binMidpoint]
#     xs = np.array(xs)
#     ys = gap + np.arange(len(coalRates))
#     plt.xticks(xticks, xs[xticks], fontsize=6)
#     plt.yticks(yticks, ys[yticks])

#     # compute MAP
#     map = np.argmax(postprob, axis=0)
#     plt.plot(np.arange(nbins), map, color='red', linestyle='--', label='MAP')
#     # compute posterior mean
#     mean = np.sum(postprob*(ys.reshape(len(coalRates), 1)), axis=0)
#     plt.plot(np.arange(nbins), mean - gap, color='orange', linestyle='--', label='Posterior Mean')

#     plt.legend(loc='lower right')
#     plt.savefig(outFigPrefix + '_pdf.pdf', dpi=300)
#     plt.clf()

#     ##### plot cdf of posterior TMRCA
#     mat_cdf = np.apply_along_axis(np.cumsum, 0, postprob)

#     # 25% percentile curve    
#     index_low = np.apply_along_axis(find_nearest, 0, mat_cdf, 0.025)
#     index_high = np.apply_along_axis(find_nearest, 0, mat_cdf, 0.975)

#     plt.imshow(mat_cdf, cmap='viridis', aspect='auto')
#     plt.colorbar()
#     plt.plot(np.arange(nbins), index_low, color='red', label='$2.5\%$ and $97.5\%$ percentile')
#     plt.plot(np.arange(nbins), index_high, color='red')
#     plt.xlabel('Segment Length (cM)')
#     plt.ylabel('TMRCA (generations backward in time)')
#     plt.legend(loc='lower right')
#     plt.xticks(xticks, xs[xticks], fontsize=6)
#     plt.yticks(yticks, ys[yticks])
#     plt.savefig(outFigPrefix + '_cdf.pdf', dpi=300)
#     plt.clf()

#     return postprob


def plot2PopIMfit(coalRates, ibds_by_chr, ch_len_dict, nPairs, time1, time2, timeBound=None, minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, step=0.25, FP=None, R=None, POWER=None, outFolder="", prefix=""):
    bins_infer = np.arange(minL_infer, maxL_infer+step, step)
    binMidpoint_infer = (bins_infer[1:] + bins_infer[:-1])/2

    outFigPrefix = f"{outFolder}/postTMRCA"
    if len(prefix) > 0:
        outFigPrefix = outFigPrefix + "." + prefix

    G = np.array([v for k, v in ch_len_dict.items()])
    if timeBound == None:
        meanNumIBD_theory, _ = twoPopIM_given_coalRate(coalRates, G, binMidpoint_infer, abs(time1 - time2))
    else:
        low1 = time1 + timeBound[0][0]
        low2 = time2 + timeBound[1][0]
        startingTime = max(low1, low2) # this is the most recent time point for which pop1 and pop2 are temporally overlapping, so this is the time point from which cross-coalescence rate will be estimated backward in time
        time1 -= startingTime
        time2 -= startingTime # normalize with respect to starting time. this should make it easier to index into coalRates vector.
        meanNumIBD_theory = np.zeros(len(binMidpoint_infer))
        weight_per_combo = 1/((1 + timeBound[0][1] - timeBound[0][0])*(1 + timeBound[1][1] - timeBound[1][0]))
        for i in np.arange(time1 + timeBound[0][0], time1 + timeBound[0][1] + 1):
            for j in np.arange(time2 + timeBound[1][0], time2 + timeBound[1][1] + 1):
                assert(max(i,j) >= 0)
                lambdas_, _ = twoPopIM_given_coalRate(coalRates[max(i,j):], G, binMidpoint_infer, abs(i - j))
                meanNumIBD_theory += weight_per_combo*lambdas_

    meanNumIBD_theory = 4*(step/100)*meanNumIBD_theory
    plt.plot(binMidpoint_infer, meanNumIBD_theory, color='red', label='IBD from inferred coal rates')

    if (not FP is None) and (not R is None) and (not POWER is None):
        bins_calc = np.arange(minL_calc, maxL_calc+step, step)
        binMidpoint = (bins_calc[1:]+bins_calc[:-1])/2
        s = np.where(np.isclose(binMidpoint, binMidpoint_infer[0]))[0][0]
        e = np.where(np.isclose(binMidpoint, binMidpoint_infer[-1]))[0][0]
        if timeBound == None:
            meanNumIBD_theory, _ = twoPopIM_given_coalRate_withError(coalRates, G, binMidpoint, abs(time1 - time2), FP=FP, R=R, POWER=POWER)
        else:
            low1 = time1 + timeBound[0][0]
            low2 = time2 + timeBound[1][0]
            startingTime = max(low1, low2) # this is the most recent time point for which pop1 and pop2 are temporally overlapping, so this is the time point from which cross-coalescence rate will be estimated backward in time
            time1 -= startingTime
            time2 -= startingTime # normalize with respect to starting time. this should make it easier to index into coalRates vector.
            meanNumIBD_theory = np.zeros(len(binMidpoint_infer))
            weight_per_combo = 1/((1 + timeBound[0][1] - timeBound[0][0])*(1 + timeBound[1][1] - timeBound[1][0]))
            for i in np.arange(time1 + timeBound[0][0], time1 + timeBound[0][1] + 1):
                for j in np.arange(time2 + timeBound[1][0], time2 + timeBound[1][1] + 1):
                    assert(max(i,j) >= 0)
                    lambdas_, _ = twoPopIM_given_coalRate_withError(coalRates[max(i,j):], G, binMidpoint, abs(i - j), FP=FP, R=R, POWER=POWER)
                    meanNumIBD_theory += weight_per_combo*lambdas_
        meanNumIBD_theory = meanNumIBD_theory[s:e+1]
        meanNumIBD_theory = 4*(step/100)*meanNumIBD_theory
        plt.plot(binMidpoint_infer, meanNumIBD_theory, color='orange', linestyle='--', label='IBD from inferred coal rates (error model correction)')

    ibds_all = []
    for ch, ibds in ibds_by_chr.items():
        ibds_all.extend(ibds)
    x, _ = np.histogram(ibds_all, bins=bins_infer)
    plt.scatter(binMidpoint_infer, x/nPairs, s=7.0, label='simulated IBD', color='grey')

    plt.xlabel('Segment Length (cM)')
    plt.ylabel('Number of Segments per Pair of Samples')
    plt.legend(loc='upper right', fontsize='medium')
    plt.yscale('log')
    plt.savefig(outFigPrefix + '.fit.pdf', dpi=300)
    plt.clf()







def plot_pairwise_TMRCA(dates, estNe, Tmax, outFolder, prefix="pairwise", minL=6.0, maxL=20.0, step=0.1):

    bins = np.arange(minL, maxL+step, step)
    binMidpoint = (bins[1:]+bins[:-1])/2
    popLabels = dates.keys()
    n = len(popLabels)
    numSubplot = int(n*(n+1)/2)
    # let's say we fix the number of columns to be 5
    ncol = 5
    nrow = math.ceil(numSubplot/5)
    fig = plt.figure(tight_layout=True, figsize=(20, 5*nrow))
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.3, hspace=0.3)

    offset = np.inf
    for k, time in dates.items():
        if time < offset:
            offset = time

    # set up the canvas to plot cdf with 2.5 and 97.5 percentile curve
    index = 0
    for id1, id2 in itertools.combinations_with_replacement(popLabels, 2):
        id1, id2 = min(id1, id2), max(id1, id2)
        i, j = index//ncol, index%ncol
        ax = fig.add_subplot(gs[i,j])
        
        index += 1

        s = max(dates[id1], dates[id2]) - offset
        postprob = computePosteriorTMRCA(1/(2*estNe[s:s+Tmax]), binMidpoint, abs(dates[id1] - dates[id2]))
        mat_cdf = np.apply_along_axis(np.cumsum, 0, postprob)
        ax.imshow(mat_cdf, cmap='viridis', aspect='auto')        

        bins = np.arange(minL, maxL+step, step)
        binMidpoint = (bins[1:]+bins[:-1])/2
        yticks = np.arange(0, Tmax, 10)
        ys = s + offset + np.arange(Tmax)
        nbins =len(binMidpoint)
        ax.set_xticks(closest_indices(binMidpoint, np.arange(minL, maxL, 2)))
        ax.set_xticklabels(map(int, np.arange(minL, maxL, 2)))
        ax.set_yticks(yticks)
        ax.set_yticklabels(ys[yticks])

        index_low = np.apply_along_axis(find_nearest, 0, mat_cdf, 0.025)
        index_high = np.apply_along_axis(find_nearest, 0, mat_cdf, 0.975)
        ax.plot(np.arange(nbins), index_low, color='red', linewidth=2, label='$2.5\%$ and \n$97.5\%$ \npercentile')
        ax.plot(np.arange(nbins), index_high, color='red')
        if i == j == 0:
            ax.legend(loc='upper right', fontsize=16)
        title = f'({dates[id1]}, {dates[id2]})' if id1 != id2 else f'{dates[id1]}'
        ax.set_title(title, fontsize=16)
        ax.tick_params(labelsize=14)
        y_cutoff = np.apply_along_axis(find_nearest, 0, mat_cdf, 0.99)
        ax.set_ylim(0, np.max(y_cutoff))

    
    fig.text(0.5, 0.05, 'IBD Segment Length (cM)', ha='center', va='center', fontsize=20)
    fig.text(0.075, 0.5, 'Generations Backward in Time', ha='center', va='center', rotation='vertical', fontsize=20)
    # add color bar
    plt.colorbar(ax.imshow(mat_cdf, cmap='viridis', aspect='auto'))
    plt.gcf().set_facecolor('white')
    plt.savefig(f'{outFolder}/{prefix}.postTMRCA_cdf.png', dpi=300, bbox_inches = "tight")
    plt.savefig(f'{outFolder}/{prefix}.postTMRCA_cdf.pdf', bbox_inches = "tight")
    plt.clf()

    # # set up the canvas to plot pdf 
    # index = 0
    # for id1, id2 in itertools.combinations_with_replacement(popLabels, 2):
    #     id1, id2 = min(id1, id2), max(id1, id2)
    #     i, j = index//ncol, index%ncol
    #     ax = fig.add_subplot(gs[i,j])
        
    #     index += 1        
    #     s = max(dates[id1], dates[id2]) - offset
    #     postprob = computePosteriorTMRCA(1/(2*estNe[s:s+Tmax]), binMidpoint, abs(dates[id1] - dates[id2]))
    #     ax.imshow(postprob, cmap='viridis', aspect='auto')

    #     bins = np.arange(minL, maxL+step, step)
    #     binMidpoint = (bins[1:]+bins[:-1])/2
    #     xticks = np.arange(0, len(binMidpoint), 15)
    #     yticks = np.arange(0, Tmax, 10)
    #     xs = [round(x, 3) for x in binMidpoint]
    #     xs = np.array(xs)
    #     ys = s + offset + np.arange(Tmax)
    #     nbins = len(binMidpoint)
    #     ax.set_xticks(xticks)
    #     ax.set_xticklabels(xs[xticks], fontsize=6)
    #     ax.set_yticks(yticks)
    #     ax.set_yticklabels(ys[yticks], fontsize=6)

    #     # compute MAP
    #     max_a_post = np.argmax(postprob, axis=0)
    #     ax.plot(np.arange(nbins), max_a_post, color='red', linestyle='--', linewidth=0.8, label='MAP')
    #     # compute posterior mean
    #     mean = np.sum(postprob*(ys.reshape(Tmax, 1)), axis=0)
    #     ax.plot(np.arange(nbins), mean - s - offset, color='orange', linestyle='--', linewidth=0.8, label='Posterior Mean')
    #     if i == j == 0:
    #         ax.legend(loc='lower right', fontsize=4)

    #     title = f'({dates[id1]}, {dates[id2]})' if id1 != id2 else f'{dates[id1]}'
    #     ax.set_title(title, fontsize=4)
    #     ax.tick_params(labelsize=4)
    
    # fig.text(0.5, 0.05, 'IBD Segment Length (cM)', ha='center', va='center', fontsize=8)
    # fig.text(0.075, 0.5, 'Generations Backward in Time', ha='center', va='center', rotation='vertical', fontsize=8)
    # plt.gcf().set_facecolor('white')
    # plt.savefig(f'{outFolder}/{prefix}.postTMRCA_pdf.png', dpi=300, bbox_inches = "tight")
    # plt.savefig(f'{outFolder}/{prefix}.postTMRCA_pdf.pdf', bbox_inches = "tight")
    # plt.clf()