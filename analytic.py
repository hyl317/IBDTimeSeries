import numpy as np
import math
from scipy.optimize import minimize
from numba import jit
from scipy.special import logsumexp


def singlePop_2tp(G, L, T, N):
    # calculate expected number of IBD segments of length L
    # for a chromosome of length G,
    # in a single well-mixed population with time-stratified sampling, constant Ne, time gap given by T, meausred in generations
    part1 = np.exp(-L*T)/((1+4*N*L)**3)
    part2 = 8*N*(1+4*N*G) + 2*T*(1+4*G*N)*(1+4*N*L) + (G-L)*(T+4*N*L*T)**2
    return part1*part2

def singlePop_2tp_given_Ne(Ne, G, L, gap):
    # G: a list of chromosome lengths, given in cM
    accu = np.zeros(len(L))
    L = L/100
    for chrLen in G/100:
        accu += singlePop_2tp(chrLen, L, gap, Ne)
    return accu

# @jit(nopython=True)
# def singlePop_2tp_integral_from_a(G, L, T, N, a):
#     part1 = -np.exp((-a*(1 + 4*N*L) + T + 2*L*N*T)/(2*N))/((1 + 4*N*L)**3)
#     part2 = -8*N*(1 + 4*G*N) - 4*(G - L)*(a + 4*a*L*N)**2 + 2*T*(1 + 4*G*N)*(1 + 4*N*L) - (G - L)*(T + 4*N*L*T)**2 + 4*a*(1 + 4*N*L)*(-1 - 4*G*N + T*(G - L)*(1 + 4*N*L))
#     return part1*part2


@jit(nopython=True)
def singlePop_2tp_given_vecNe(Ne, G, L, gap, tail=False):
    # Ne is a vector, which gives a trajectory of pop size over the past generations
    # G is a vector of chromosome length, length given in cM
    # L is also a vector, length of IBD segments (given in cM)
    # T is the sampling gap time
    # FP, R, POWER are error models
    # FP: a vector of the same length as L, which describes false positive rate of segments of certain length
    # R: a matrix with len(L) rows and len(L) columns, R[i,j] describes the probability of inferring a segment of length L[i] to be of length L[j]
    # POWER: a vector of the same length as L, the power of detecting segments of certain length
    # return a vector of the same length as L
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
    EK = np.zeros_like(mat1) # this is for gradient calculation

    ####################### original implementation ##############
    for chrLen in G/100:
        # sum over from t=gap+1 to t=gap+len(Ne)
        mat = mat1 + (((2*Tvec - gap)**2)@(chrLen - Lvec))*common_part
        EK += mat
        mat = coalProb*mat
        accu += np.sum(mat, axis=0)
        # if tail:
        #     accu += singlePop_2tp_integral_from_a(chrLen, Lvec, gap, Ne[-1], len(Ne)).flatten()

    ################### calculate gradient ########################
    # grad[i,j] = gradient of lambda_j with respect to N_i
    # cumprod = np.exp(tmp[:-1])
    # grad = np.zeros((nGen, nBins))
    # for i in np.arange(nGen):
    #     part1 = (-1/(2*Ne[i]**2))*cumprod[i]
    #     part2 = (1/(2*Ne[i]**2))*(1/(2*Ne[i+1:]))*(cumprod[i+1:]/(1-1/(2*Ne[i])))
    #     for j in np.arange(nBins):
    #         grad[i,j] = part1*EK[i,j] + np.sum(part2*EK[i+1:,j])

    cumprod_logsum = tmp[:-1]
    grad = np.zeros((nGen, nBins))
    for i in np.arange(nGen):
        part1 = -np.log(2) -2*np.log(Ne[i]) + cumprod_logsum[i]
        part1 = -np.exp(part1)
        part2 = -np.log(2) -2*np.log(Ne[i]) - np.log(2*Ne[i+1:]) + cumprod_logsum[i+1:] - np.log(1-1/(2*Ne[i]))
        part2 = np.exp(part2)
        for j in np.arange(nBins):
            grad[i,j] = part1*EK[i,j] + np.sum(part2*EK[i+1:,j])

    return accu, grad

@jit(nopython=True)
def singlePop_2tp_given_vecNe_withError(Ne, G, L, gap, FP, R, POWER):
    # Ne is a vector, which gives a trajectory of pop size over the past generations
    # G is a vector of chromosome length, length given in cM
    # L is also a vector, length of IBD segments (given in cM)
    # T is the sampling gap time
    # FP, R, POWER are error models
    # FP: a vector of the same length as L, which describes false positive rate of segments of certain length per centiMorgan per haplotype pair
    # R: a matrix with len(L) rows and len(L) columns, R[i,j] describes the probability of inferring a segment of length L[i] to be of length L[j]
    # POWER: a vector of the same length as L, the power of detecting segments of certain length
    # return a vector of the same length as L
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
    EK = np.zeros_like(mat1) # this is for gradient calculation

    ####################### original implementation ##############
    for chrLen in G/100:
        # sum over from t=gap+1 to t=gap+len(Ne)
        mat = mat1 + (((2*Tvec - gap)**2)@(chrLen - Lvec))*common_part
        EK += mat
        mat = coalProb*mat
        accu += np.sum(mat, axis=0)

    cumprod_logsum = tmp[:-1]
    grad = np.zeros((nGen, nBins))
    for i in np.arange(nGen):
        part1 = -np.log(2) -2*np.log(Ne[i]) + cumprod_logsum[i]
        part1 = -np.exp(part1)
        part2 = -np.log(2) -2*np.log(Ne[i]) - np.log(2*Ne[i+1:]) + cumprod_logsum[i+1:] - np.log(1-1/(2*Ne[i]))
        part2 = np.exp(part2)
        for j in np.arange(nBins):
            grad[i,j] = part1*EK[i,j] + np.sum(part2*EK[i+1:,j])

    # update lambda and gradient according to the given error model
    # evaluate the integral by summing over bins
    # print(f'length of power: {POWER.shape}')
    # print(f'shape of accu: {accu.shape}')
    detected = POWER*accu
    inferred = np.sum(R*(detected.reshape(nBins, 1)), axis=0) # elementwise mul and then sum over column
    accu = FP*np.sum(G)*100 + inferred # update accu to incorporate error models, need to multiply by 100 cuz we wanna convert FP rate from per centiMorgan to per Morgan

    # now update the gradient
    grad_new = np.zeros_like(grad)
    for i in np.arange(nBins):
        grad_new[:, i] = np.sum(grad*POWER*(R[:, i].copy().reshape(1, nBins)), axis=1) # sum over rows
    grad = grad_new

    return accu, grad



def singlePop_2tp_given_Ne_negLoglik(Ne, histogram, binMidpoint, G, gap, numPairs):
    assert(len(histogram) == len(binMidpoint))
    step = binMidpoint[1] - binMidpoint[0]
    lambdas = singlePop_2tp_given_Ne(Ne, G, binMidpoint, gap)
    lambdas = lambdas*numPairs*(step/100)
    loglik_each_bin = histogram*np.log(lambdas) - lambdas
    return -np.sum(loglik_each_bin)

def singlePop_2tp_given_vecNe_negLoglik_noPenalty(Ne, histogram, binMidpoint, G, gap, numPairs, s=0, e=-1, FP=None, R=None, POWER=None, tail=False):
    # G: chromosome length, given in cM
    # binMidPoint: given in cM
    assert(len(histogram) == len(binMidpoint))
    step = binMidpoint[1] - binMidpoint[0]

    if (FP is None) or (R is None) or (POWER is None):
        lambdas, grad_mat = singlePop_2tp_given_vecNe(Ne, G, binMidpoint, gap, tail=tail)
    else:
        lambdas, grad_mat = singlePop_2tp_given_vecNe_withError(Ne, G, binMidpoint, gap, FP, R, POWER)
    # subset to segment length range of interest for inference
    histogram = histogram[s:e]
    lambdas = lambdas[s:e]
    grad_mat = grad_mat[:, s:e]
    lambdas = lambdas*numPairs*(step/100)
    loglik_each_bin = histogram*np.log(lambdas) - lambdas
    grad = -numPairs*(step/100)*np.sum((histogram/lambdas - 1).reshape((1, len(histogram)))*grad_mat, axis=1).flatten()
    return -np.sum(loglik_each_bin), grad


################################################### code for estimating cross coal rate #########################################################
#################################################################################################################################################
#################################################################################################################################################

def twoPopIM_2tp(G, L, T, coalrate):
    # calculate expected number of IBD segments of length L
    # for a chromosome of length G,
    # in a single well-mixed population with time-stratified sampling, constant coalescent rate, time gap given by T, meausred in generations
    N = 1/(2*coalrate)
    part1 = np.exp(-L*T)/((1+4*N*L)**3)
    part2 = 8*N*(1+4*N*G) + 2*T*(1+4*G*N)*(1+4*N*L) + (G-L)*(T+4*N*L*T)**2
    return part1*part2

def twoPopIM_2tp_given_coalrate(coalrate, G, L, gap):
    # G: a list of chromosome lengths, given in cM
    accu = np.zeros(len(L))
    L = L/100
    for chrLen in G/100:
        accu += twoPopIM_2tp(chrLen, L, gap, coalrate)
    return accu

def twoPopIM_2tp_given_coalrate_negLoglik(coalrate, histogram, binMidpoint, G, gap, numPairs):
    assert(len(histogram) == len(binMidpoint))
    step = binMidpoint[1] - binMidpoint[0]
    lambdas = twoPopIM_2tp_given_coalrate(coalrate, G, binMidpoint, gap)
    lambdas = lambdas*numPairs*(step/100)
    loglik_each_bin = histogram*np.log(lambdas) - lambdas
    return -np.sum(loglik_each_bin)

@jit(nopython=True)
def twoPopIM_given_coalRate(coalRates, G, L, timeGap):
    """
    Given coalescent rates, return the expected IBD sharing at length bin given by L
    """
    nBins = len(L)
    nGen = len(coalRates)
    accu = np.zeros(nBins)
    Lvec = np.reshape(L, (1, nBins))/100 # transform to row vector
    Tvec = np.reshape(timeGap + np.arange(1.0, nGen+1.0), (nGen, 1)) # transform to column vector
    tmp = np.zeros(nGen+1)
    tmp[1:] = np.log(1 - coalRates)
    tmp = np.cumsum(tmp)
    coalProb = tmp[:-1] + np.log(coalRates)
    coalProb = np.exp(coalProb)
    coalProb = np.reshape(coalProb, (len(coalProb), 1)) # transform to column vector
    
    common_part = np.exp(-(2*Tvec - timeGap)@Lvec)
    mat1 = 2*(2*Tvec - timeGap)*common_part
    EK = np.zeros_like(mat1) # this is for gradient calculation

    ####################### original implementation ##############
    for chrLen in G/100:
        # sum over from t=gap+1 to t=gap+len(Ne)
        mat = mat1 + (((2*Tvec - timeGap)**2)@(chrLen - Lvec))*common_part
        EK += mat
        mat = coalProb*mat
        accu += np.sum(mat, axis=0)

    cumprod_logsum = tmp[:-1]
    grad = np.zeros((nGen, nBins))
    for i in np.arange(nGen):
        part1 = np.exp(cumprod_logsum[i])
        part2 = np.log(coalRates[i+1:]) + cumprod_logsum[i+1:] - np.log(1 - coalRates[i])
        part2 = -np.exp(part2)
        for j in np.arange(nBins):
            grad[i,j] = part1*EK[i,j] + np.sum(part2*EK[i+1:,j])
    
    return accu, grad


@jit(nopython=True)
def twoPopIM_given_coalRate_withError(coalRates, G, L, timeGap, FP, R, POWER):
    """
    Given coalescent rates, return the expected IBD sharing at length bin given by L
    """
    nBins = len(L)
    nGen = len(coalRates)
    accu = np.zeros(nBins)
    Lvec = np.reshape(L, (1, nBins))/100 # transform to row vector
    Tvec = np.reshape(timeGap + np.arange(1.0, nGen+1.0), (nGen, 1)) # transform to column vector
    tmp = np.zeros(nGen+1)
    tmp[1:] = np.log(1 - coalRates)
    tmp = np.cumsum(tmp)
    coalProb = tmp[:-1] + np.log(coalRates)
    coalProb = np.exp(coalProb)
    coalProb = np.reshape(coalProb, (len(coalProb), 1)) # transform to column vector
    
    common_part = np.exp(-(2*Tvec - timeGap)@Lvec)
    mat1 = 2*(2*Tvec - timeGap)*common_part
    EK = np.zeros_like(mat1) # this is for gradient calculation

    ####################### original implementation ##############
    for chrLen in G/100:
        # sum over from t=gap+1 to t=gap+len(Ne)
        mat = mat1 + (((2*Tvec - timeGap)**2)@(chrLen - Lvec))*common_part
        EK += mat
        mat = coalProb*mat
        accu += np.sum(mat, axis=0)

    cumprod_logsum = tmp[:-1]
    grad = np.zeros((nGen, nBins))
    for i in np.arange(nGen):
        part1 = np.exp(cumprod_logsum[i])
        part2 = np.log(coalRates[i+1:]) + cumprod_logsum[i+1:] - np.log(1 - coalRates[i])
        part2 = -np.exp(part2)
        for j in np.arange(nBins):
            grad[i,j] = part1*EK[i,j] + np.sum(part2*EK[i+1:,j])

    # update lambda and gradient according to the given error model
    # evaluate the integral by summing over bins
    detected = POWER*accu
    inferred = np.sum(R*(detected.reshape(nBins, 1)), axis=0) # elementwise mul and then sum over column
    accu = FP*np.sum(G)*100 + inferred # update accu to incorporate error models, need to multiply by 100 cuz we wanna convert FP rate from per centiMorgan to per Morgan

    # now update the gradient
    grad_new = np.zeros_like(grad)
    for i in np.arange(nBins):
        grad_new[:, i] = np.sum(grad*POWER*(R[:, i].copy().reshape(1, nBins)), axis=1) # sum over rows
    grad = grad_new
    
    return accu, grad

def twoPopIM_given_vecCoalRates_negLoglik_noPenalty(coalRates, histogram, binMidpoint, G, numPairs, timeGap, s=0, e=-1, FP=None, R=None, POWER=None):
    # G: chromosome length, given in cM
    # binMidPoint: given in cM
    assert(len(histogram) == len(binMidpoint))
    step = binMidpoint[1] - binMidpoint[0]
    if (FP is None) or (R is None) or (POWER is None):
        lambdas, grad_mat = twoPopIM_given_coalRate(coalRates, G, binMidpoint, timeGap)
    else:
        lambdas, grad_mat = twoPopIM_given_coalRate_withError(coalRates, G, binMidpoint, timeGap, FP, R, POWER)
    # subset to segment length range of interest for inference
    histogram = histogram[s:e]
    lambdas = lambdas[s:e]
    grad_mat = grad_mat[:, s:e]
    lambdas = lambdas*numPairs*(step/100)
    loglik_each_bin = histogram*np.log(lambdas) - lambdas
    grad = -numPairs*(step/100)*np.sum((histogram/lambdas - 1).reshape((1, len(histogram)))*grad_mat, axis=1).flatten()
    return -np.sum(loglik_each_bin), grad

def computePosteriorTMRCA(coalRates, l, gap):
    """
    Compute posterior TMRCA distribution for segments of length l, given a vector of estimated coalescent rates over generations
    l: a vector of length of IBD segments, given in unit of cM.
    gap: sampling time gap of the two haplotypes
    coalRates: coalescent rates between the two haplotypes, starting from the older of the two haplotypes backward in time.
    Return: a matrix of dimension (len(coalRates), len(l)). Mat[i,j] gives the posterior probability of segment of length l[j] having TMRCA at i generations backward in time, given the estimated coalescent rate trajectory.
    """
    numG = len(coalRates)
    log_not_yet_coalesced = np.zeros_like(coalRates)
    log_not_yet_coalesced[1:] = np.cumsum(np.log(1 - coalRates))[:-1]
    log_tmrca_given_coal = np.log(coalRates) + log_not_yet_coalesced

    g = gap + np.arange(1, 1+numG)
    g = g.reshape(numG, 1) # reshape to column vector
    l = l.reshape(1, len(l)) # reshape to row vector
    log_len_given_tmrca = 2*np.log((2*g-gap)/100) + np.log(l) - (2*g-gap)*l/100
    
    log_post = log_len_given_tmrca + log_tmrca_given_coal.reshape(numG, 1)
    normalizing_const = np.apply_along_axis(logsumexp, 0, log_post)
    post_prob = np.exp(log_post - normalizing_const)
    assert(np.all(np.isclose(np.sum(post_prob, axis=0), 1.0)))
    return post_prob