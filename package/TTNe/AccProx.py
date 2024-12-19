from numba import jit
import itertools
import numpy as np
import optimization as opt
import mushi.optimization
from analytic import singlePop_2tp_given_vecNe

@jit(nopython=True)
def singlePop_2tp_given_vecNe_lambda(Ne, G, L, gap):
    # same as singlePop_2tp_given_vecNe in the analytic.py module
    # but this one only returns the lambda vector. It doesn't compute the gradient at the same time
    nBins = len(L)
    nGen = len(Ne)
    accu = np.zeros(nBins)
    Lvec = np.reshape(L, (1, nBins))/100 # transform to row vector
    Tvec = np.reshape(gap + np.arange(1.0, len(Ne)+1.0), (nGen, 1)) # transform to column vector
    tmp = np.zeros(nGen+1)
    tmp[1:] = np.log(1-1/(2*Ne))
    tmp = np.cumsum(tmp)
    coalProb = tmp[:-1] + np.log(1/(2*Ne))
    coalProb = np.exp(coalProb)
    coalProb = np.reshape(coalProb, (len(coalProb), 1)) # transform to column vector

    common_part = np.exp(-(2*Tvec - gap)@Lvec)
    mat1 = 2*(2*Tvec - gap)*common_part

    for chrLen in G/100:
        # sum over from t=gap+1 to t=gap+len(Ne)
        mat = mat1 + (((2*Tvec - gap)**2)@(chrLen - Lvec))*common_part
        mat = coalProb*mat
        accu += np.sum(mat, axis=0)
    return accu


@jit(nopython=True)
def singlePop_2tp_given_vecNe_grad(Ne, G, L, gap):
    # same as singlePop_2tp_given_vecNe in the analytic.py module
    # but this one only returns the gradient matrix (lambda_i's derivative with respect to N_j)
    nBins = len(L)
    nGen = len(Ne)
    Lvec = np.reshape(L, (1, nBins))/100 # transform to row vector
    Tvec = np.reshape(gap + np.arange(1.0, len(Ne)+1.0), (nGen, 1)) # transform to column vector
    tmp = np.zeros(nGen+1)
    tmp[1:] = np.log(1-1/(2*Ne))
    tmp = np.cumsum(tmp)
    common_part = np.exp(-(2*Tvec - gap)@Lvec)
    mat1 = 2*(2*Tvec - gap)*common_part
    EK = np.zeros_like(mat1) # this is for gradient calculation

    for chrLen in G/100:
        # sum over from t=gap+1 to t=gap+len(Ne)
        mat = mat1 + (((2*Tvec - gap)**2)@(chrLen - Lvec))*common_part
        EK += mat
 
    cumprod_logsum = tmp[:-1]
    grad = np.zeros((nGen, nBins))
    for i in np.arange(nGen):
        part1 = -np.log(2) -2*np.log(Ne[i]) + cumprod_logsum[i]
        part1 = -np.exp(part1)
        part2 = -np.log(2) -2*np.log(Ne[i]) - np.log(2*Ne[i+1:]) + cumprod_logsum[i+1:] - np.log(1-1/(2*Ne[i]))
        part2 = np.exp(part2)
        for j in np.arange(nBins):
            grad[i,j] = part1*EK[i,j] + np.sum(part2*EK[i+1:,j])

    return grad

def singlePop_2tp_given_vecNe_negLoglik_noPenalty_g(Ne, histogram, binMidpoint, G, gap, numPairs):
    # G: chromosome length, given in cM
    # binMidPoint: given in cM
    assert(len(histogram) == len(binMidpoint))
    step = binMidpoint[1] - binMidpoint[0]
    lambdas = singlePop_2tp_given_vecNe_lambda(Ne, G, binMidpoint, gap)
    lambdas = lambdas*numPairs*(step/100)
    loglik_each_bin = histogram*np.log(lambdas) - lambdas
    return -np.sum(loglik_each_bin)

def singlePop_2tp_given_vecNe_negLoglik_noPenalty_grad(Ne, histogram, binMidpoint, G, gap, numPairs):
    # G: chromosome length, given in cM
    # binMidPoint: given in cM
    assert(len(histogram) == len(binMidpoint))
    step = binMidpoint[1] - binMidpoint[0]
    lambdas, grad_mat = singlePop_2tp_given_vecNe(Ne, G, binMidpoint, gap)
    lambdas = lambdas*numPairs*(step/100)
    grad = -numPairs*(step/100)*np.sum((histogram/lambdas - 1).reshape((1, len(histogram)))*grad_mat, axis=1).flatten()
    return grad

def g(Ne, histograms, binMidpoint, G, gaps, nHaplotypePairs, time_offset):
    # return the negative loglikelihood of Ne, plus the differentiable penalty term
    # G: a vector of chromosome length
    # gaps: dictionary, where the key is the sampling cluster index and the value is the sampling time
    # nSamples: dictionary, where the key is the sampling cluster index and the value is the number of diploid samples within each cluster
    Ne = np.exp(Ne)
    accu = 0
    for (id1, id2), nHaplotypePair in nHaplotypePairs.items():
        if nHaplotypePair == 0:
            continue
        accu += singlePop_2tp_given_vecNe_negLoglik_noPenalty_g(Ne[max(gaps[id1], gaps[id2])-time_offset:], histograms[(min(id1, id2), max(id1, id2))], \
            binMidpoint, G, abs(gaps[id1] - gaps[id2]), nHaplotypePair)
    
    #accu += beta*np.sum((np.log(Ne)-np.log(Nconst))**2)
    return accu

def grad(Ne, histograms, binMidpoint, G, gaps, nHaplotypePairs, time_offset):
    # gradient function of the above function g()
    Ne = np.exp(Ne)
    grad = np.zeros_like(Ne)
    for (id1, id2), nHaplotypePair in nHaplotypePairs.items():
        if nHaplotypePair == 0:
            continue
        grad[max(gaps[id1], gaps[id2])-time_offset:] += singlePop_2tp_given_vecNe_negLoglik_noPenalty_grad(Ne[max(gaps[id1], gaps[id2])-time_offset:], histograms[(min(id1, id2), max(id1, id2))], \
            binMidpoint, G, abs(gaps[id1] - gaps[id2]), nHaplotypePair)
    
    #grad += beta*np.sum(2*(np.log(Ne) - np.log(Nconst))/Ne)
    return grad




def AccProx_trendFiltering_l1(Ne, histograms, binMidpoint, G, gaps, nHaplotypePairs, Tmax, alpha, beta, timeBoundDict, Nconst, s=0, e=-1, FP=None, R=None, POWER=None):
    # optimize the objective function using the Nesterov accelerated proximal gradient method implemented in MuShi
    initNe = np.log(Ne)
    params = [histograms, binMidpoint, G, gaps, nHaplotypePairs, 0]
    ########################### auxiliary function defined inside this big function ##############################
    # defined as inner function so that I don't need to include alpha in params
    @jit(nopython=True)
    def l1_penalty(Ne):
        return alpha*np.sum(np.abs(np.diff(Ne, n=2))) + beta*np.sum(np.abs(np.diff(Ne, n=1)))
    def prox(Ne, s):
        trend_filter = mushi.optimization.TrendFilter((1,0), (s*alpha,s*beta))
        return trend_filter.run(Ne)
    ###############################################################################################################

    optimizer = opt.AccProxGrad(g, grad, l1_penalty, prox, params, verbose=True)
    return np.exp(optimizer.run(initNe))