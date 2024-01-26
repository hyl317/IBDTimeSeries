
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


