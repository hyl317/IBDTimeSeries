# some plotting function for ease of visualizing and interpreting the results

import math
import matplotlib.gridspec as gridspec
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from analytic import singlePop_2tp_given_vecNe, singlePop_2tp_given_vecNe_withError, computePosteriorTMRCA, twoPopIM_given_coalRate, twoPopIM_given_coalRate_withError

def plot_pairwise_fitting(ibds, gaps, nSamples, ch_len_dict, estNe, outFolder, prefix="pairwise", \
        minL_calc=2.0, maxL_calc=24, minL_infer=6.0, maxL=20.0, step=0.25, minL_plot=6.0, FP=None, R=None, POWER=None):

    bins = np.arange(minL_plot, maxL+step, step)
    midpoint = (bins[1:]+bins[:-1])/2
    L = np.array([v for _, v in ch_len_dict.items()])

    popLabels = nSamples.keys()
    n = len(popLabels)
    numSubplot = int(n*(n+1)/2)
    # let's say we fix the number of columns to be 5
    ncol = 5
    nrow = math.ceil(numSubplot/5)
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.15, hspace=0.3)

    offset = np.inf
    for k, time in gaps.items():
        if time < offset:
            offset = time

    # set up the canvas
    index = 0
    for id1, id2 in itertools.combinations_with_replacement(popLabels, 2):
        id1, id2 = min(id1, id2), max(id1, id2)
        ibd_simulated = []
        for ch in ch_len_dict.keys():
            if ibds[(id1, id2)].get(ch):
                ibd_simulated.extend(ibds[(id1, id2)][ch])
        
        # plot the simulated/observed IBD counts
        n1, n2 = nSamples[id1], nSamples[id2]
        npairs = n1*n2 if id1 != id2 else n1*(n1-1)/2
        x, _ = np.histogram(ibd_simulated, bins=bins)
        x = np.array(x)/npairs
        
        i, j = index//ncol, index%ncol
        ax = fig.add_subplot(gs[i,j])
        
        index += 1
        ax.scatter(midpoint, x, label='Observed IBD sharing', s=3.0, color='grey')

        # plot fit from estimated Ne
        meanNumIBD_expectation, _ = singlePop_2tp_given_vecNe(estNe[max(gaps[id1], gaps[id2]) - offset:], L, midpoint, abs(gaps[id1]-gaps[id2]))
        meanNumIBD_expectation = 4*(step/100)*meanNumIBD_expectation
        ax.plot(midpoint, meanNumIBD_expectation, label='Expected IBD sharing from inferred Ne', color='red', linewidth=0.75)

        if (not FP is None) and (not R is None) and (not POWER is None):
            bins_calc = np.arange(minL_calc, maxL_calc+step, step)
            binMidpoint = (bins_calc[1:] + bins_calc[:-1])/2
            s = np.where(np.isclose(binMidpoint, midpoint[0]))[0][0]
            e = np.where(np.isclose(binMidpoint, midpoint[-1]))[0][0]
            meanNumIBD_theory, _ = singlePop_2tp_given_vecNe_withError(estNe[max(gaps[id1], gaps[id2]) - offset:], L, binMidpoint, abs(gaps[id1]-gaps[id2]), FP, R, POWER)
            meanNumIBD_theory = meanNumIBD_theory[s:e+1]
            meanNumIBD_theory = 4*(step/100)*meanNumIBD_theory
            ax.plot(midpoint, meanNumIBD_theory, color='orange', label='Expected IBD sharing from inferred Ne (with error correction)', linewidth=0.75)


        ##### plot a dashed vertical line to indicate which subset of IBD is used in inference
        ax.axvline(minL_infer, 0, 1, color='k', linestyle='--', linewidth=0.5)

        ax.set_yscale('log')
        title = f'({id1}:{gaps[id1]}, {id2}:{gaps[id2]})' if id1 != id2 else f'{id1}:{gaps[id1]}'
        ax.set_title(title, fontsize=4)
        ax.tick_params(labelsize=4)
    
    plt.savefig(f'{outFolder}/{prefix}.fit.png', dpi=300, bbox_inches = "tight")
    plt.savefig(f'{outFolder}/{prefix}.fit.pdf', bbox_inches = "tight")
    plt.clf()


def find_nearest(arr, val):
    # return index into arr such that the element of that index is numerically closest to val
    residual = np.abs(arr - val)
    return np.argmin(residual)



def plotPosteriorTMRCA(coalRates, gap=0, minL=6.0, maxL=20.0, step=0.25, outFolder="", prefix=""):
    """
    make a heatmap of the posterior TMRCA distribution for segments of length between minL and maxL, given a vector of estimated coalescent rates over generations.
    """
    bins = np.arange(minL, maxL+step, step)
    binMidpoint = (bins[1:] + bins[:-1])/2
    postprob = computePosteriorTMRCA(coalRates, binMidpoint, gap)

    outFigPrefix = f"{outFolder}/postTMRCA"
    if len(prefix) > 0:
        outFigPrefix = outFigPrefix + "." + prefix
    
    ##### plot pdf of posterior TMRCA
    plt.imshow(postprob, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Segment Length (cM)')
    plt.ylabel('TMRCA (generations backward in time)')
    bins = np.arange(minL, maxL+step, step)
    binMidpoint = (bins[1:]+bins[:-1])/2
    nbins =len(binMidpoint)
    xticks = np.arange(0, len(binMidpoint), 25)
    yticks = np.arange(0, len(coalRates), 10)
    xs = [round(x, 3) for x in binMidpoint]
    xs = np.array(xs)
    ys = gap + np.arange(len(coalRates))
    plt.xticks(xticks, xs[xticks], fontsize=6)
    plt.yticks(yticks, ys[yticks])

    # compute MAP
    map = np.argmax(postprob, axis=0)
    plt.plot(np.arange(nbins), map, color='red', linestyle='--', label='MAP')
    # compute posterior mean
    mean = np.sum(postprob*(ys.reshape(len(coalRates), 1)), axis=0)
    plt.plot(np.arange(nbins), mean - gap, color='orange', linestyle='--', label='Posterior Mean')

    plt.legend(loc='lower right')
    plt.savefig(outFigPrefix + '_pdf.pdf', dpi=300)
    plt.clf()

    ##### plot cdf of posterior TMRCA
    mat_cdf = np.apply_along_axis(np.cumsum, 0, postprob)

    # 25% percentile curve    
    index_low = np.apply_along_axis(find_nearest, 0, mat_cdf, 0.025)
    index_high = np.apply_along_axis(find_nearest, 0, mat_cdf, 0.975)

    plt.imshow(mat_cdf, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.plot(np.arange(nbins), index_low, color='red', label='$2.5\%$ and $97.5\%$ percentile')
    plt.plot(np.arange(nbins), index_high, color='red')
    plt.xlabel('Segment Length (cM)')
    plt.ylabel('TMRCA (generations backward in time)')
    plt.legend(loc='lower right')
    plt.xticks(xticks, xs[xticks], fontsize=6)
    plt.yticks(yticks, ys[yticks])
    plt.savefig(outFigPrefix + '_cdf.pdf', dpi=300)
    plt.clf()

    return postprob


def plot2PopIMfit(coalRates, ibds_by_chr, ch_len_dict, gap, nPairs, minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, step=0.25, FP=None, R=None, POWER=None, outFolder="", prefix=""):
    bins_infer = np.arange(minL_infer, maxL_infer+step, step)
    binMidpoint_infer = (bins_infer[1:] + bins_infer[:-1])/2

    outFigPrefix = f"{outFolder}/postTMRCA"
    if len(prefix) > 0:
        outFigPrefix = outFigPrefix + "." + prefix

    G = np.array([v for k, v in ch_len_dict.items()])
    meanNumIBD_theory, _ = twoPopIM_given_coalRate(coalRates, G, binMidpoint_infer, gap)
    meanNumIBD_theory = 4*(step/100)*meanNumIBD_theory
    plt.plot(binMidpoint_infer, meanNumIBD_theory, color='red', label='IBD from inferred coal rates')

    if (not FP is None) and (not R is None) and (not POWER is None):
        bins_calc = np.arange(minL_calc, maxL_calc+step, step)
        binMidpoint = (bins_calc[1:]+bins_calc[:-1])/2
        s = np.where(np.isclose(binMidpoint, binMidpoint_infer[0]))[0][0]
        e = np.where(np.isclose(binMidpoint, binMidpoint_infer[-1]))[0][0]
        meanNumIBD_theory, _ = twoPopIM_given_coalRate_withError(coalRates, G, binMidpoint, gap, FP=FP, R=R, POWER=POWER)
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







def plot_pairwise_TMRCA(gaps, estNe, Tmax, outFolder, prefix="pairwise", minL=6.0, maxL=20.0, step=0.1):

    bins = np.arange(minL, maxL+step, step)
    binMidpoint = (bins[1:]+bins[:-1])/2
    popLabels = gaps.keys()
    n = len(popLabels)
    numSubplot = int(n*(n+1)/2)
    # let's say we fix the number of columns to be 5
    ncol = 5
    nrow = math.ceil(numSubplot/5)
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.3, hspace=0.3)

    offset = np.inf
    for k, time in gaps.items():
        if time < offset:
            offset = time

    # set up the canvas to plot cdf with 2.5 and 97.5 percentile curve
    index = 0
    for id1, id2 in itertools.combinations_with_replacement(popLabels, 2):
        id1, id2 = min(id1, id2), max(id1, id2)
        i, j = index//ncol, index%ncol
        ax = fig.add_subplot(gs[i,j])
        
        index += 1

        s = max(gaps[id1], gaps[id2]) - offset
        postprob = computePosteriorTMRCA(1/(2*estNe[s:s+Tmax]), binMidpoint, abs(gaps[id1] - gaps[id2]))
        mat_cdf = np.apply_along_axis(np.cumsum, 0, postprob)
        ax.imshow(mat_cdf, cmap='viridis', aspect='auto')        

        bins = np.arange(minL, maxL+step, step)
        binMidpoint = (bins[1:]+bins[:-1])/2
        xticks = np.arange(0, len(binMidpoint), 15)
        yticks = np.arange(0, Tmax, 10)
        xs = [round(x, 3) for x in binMidpoint]
        xs = np.array(xs)
        ys = s + offset + np.arange(Tmax)
        nbins =len(binMidpoint)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xs[xticks], fontsize=6)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ys[yticks], fontsize=6)

        index_low = np.apply_along_axis(find_nearest, 0, mat_cdf, 0.025)
        index_high = np.apply_along_axis(find_nearest, 0, mat_cdf, 0.975)
        ax.plot(np.arange(nbins), index_low, color='red', label='$2.5\%$ and $97.5\%$ percentile')
        ax.plot(np.arange(nbins), index_high, color='red')

        title = f'({id1}:{gaps[id1]}, {id2}:{gaps[id2]})' if id1 != id2 else f'{id1}:{gaps[id1]}'
        ax.set_title(title, fontsize=4)
        ax.tick_params(labelsize=4)
    
    fig.text(0.5, 0.05, 'Segment Length (cM)', ha='center', va='center', fontsize=8)
    fig.text(0.075, 0.5, 'Generations Backward in Time', ha='center', va='center', rotation='vertical', fontsize=8)
    plt.savefig(f'{outFolder}/{prefix}.postTMRCA_cdf.png', dpi=300, bbox_inches = "tight")
    plt.savefig(f'{outFolder}/{prefix}.postTMRCA_cdf.pdf', bbox_inches = "tight")
    plt.clf()

    # set up the canvas to plot pdf 
    index = 0
    for id1, id2 in itertools.combinations_with_replacement(popLabels, 2):
        id1, id2 = min(id1, id2), max(id1, id2)
        i, j = index//ncol, index%ncol
        ax = fig.add_subplot(gs[i,j])
        
        index += 1        
        s = max(gaps[id1], gaps[id2]) - offset
        postprob = computePosteriorTMRCA(1/(2*estNe[s:s+Tmax]), binMidpoint, abs(gaps[id1] - gaps[id2]))
        ax.imshow(postprob, cmap='viridis', aspect='auto')

        bins = np.arange(minL, maxL+step, step)
        binMidpoint = (bins[1:]+bins[:-1])/2
        xticks = np.arange(0, len(binMidpoint), 15)
        yticks = np.arange(0, Tmax, 10)
        xs = [round(x, 3) for x in binMidpoint]
        xs = np.array(xs)
        ys = s + offset + np.arange(Tmax)
        nbins = len(binMidpoint)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xs[xticks], fontsize=6)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ys[yticks], fontsize=6)

        # compute MAP
        map = np.argmax(postprob, axis=0)
        ax.plot(np.arange(nbins), map, color='red', linestyle='--', linewidth=0.8, label='MAP')
        # compute posterior mean
        mean = np.sum(postprob*(ys.reshape(Tmax, 1)), axis=0)
        ax.plot(np.arange(nbins), mean - s - offset, color='orange', linestyle='--', linewidth=0.8, label='Posterior Mean')
        if i == j == 0:
            ax.legend(loc='lower right', fontsize=4)

        title = f'({id1}:{gaps[id1]}, {id2}:{gaps[id2]})' if id1 != id2 else f'{id1}:{gaps[id1]}'
        ax.set_title(title, fontsize=4)
        ax.tick_params(labelsize=4)
    
    fig.text(0.5, 0.05, 'Segment Length (cM)', ha='center', va='center', fontsize=8)
    fig.text(0.075, 0.5, 'Generations Backward in Time', ha='center', va='center', rotation='vertical', fontsize=8)
    plt.savefig(f'{outFolder}/{prefix}.postTMRCA_pdf.png', dpi=300, bbox_inches = "tight")
    plt.savefig(f'{outFolder}/{prefix}.postTMRCA_pdf.pdf', bbox_inches = "tight")
    plt.clf()