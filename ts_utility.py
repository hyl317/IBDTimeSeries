from collections import defaultdict
import msprime
import numpy as np
import multiprocessing as mp
import itertools
import time
import random
from collections import Counter

def readHapMap(path2Map):
    # assume the first row is header, so we ignore it
    bps = []
    cMs = []
    with open(path2Map) as f:
        f.readline()
        line = f.readline()
        while line:
            _, bp, _, cM = line.strip().split()
            bps.append(int(bp))
            cMs.append(float(cM))
            line = f.readline()
    return np.array(bps), np.array(cMs)


def bp2Morgan(bp, bps, cMs):
    # bps: a list of basepair position
    # cMs: a list of geneticMap position in cM corresponding to bps
    assert(len(bps) == len(cMs))
    i = np.searchsorted(bps, bp, side='left')
    if bps[i] == bp:
        return cMs[i]
    elif i == 0:
        return cMs[0]*(bp/bps[0])
    else:
        left_bp, right_bp = bps[i-1], bps[i]
        left_cM, right_cM = cMs[i-1], cMs[i]
        return left_cM + (right_cM - left_cM)*(bp - left_bp)/(right_bp - left_bp)

def ibd_segments_full_ARGs(ts, a, b, bps, cMs, maxGen=np.inf, minLen=4):
    # this is used when only 1 haplotype is sampled from each island
    segment_lens = []
    trees_iter = ts.trees()
    next(trees_iter) # I don't want the first tree because recombination map doesn't cover it
    for tree in trees_iter:
        left = bp2Morgan(tree.interval[0], bps, cMs)
        right = bp2Morgan(tree.interval[1], bps, cMs)
        if right - left >= minLen and (tree.mrca(a,b) != -1 and tree.tmrca(a,b) < maxGen):
            segment_lens.append(right - left)
    return segment_lens

def ibd_segments_full_ARGs_plusTMRCA(ts, a, b, bps, cMs, maxGen=np.inf, minLen=4):
    # this is used when only 1 haplotype is sampled from each island
    segment_lens = []
    trees_iter = ts.trees()
    next(trees_iter) # I don't want the first tree because recombination map doesn't cover it
    for tree in trees_iter:
        left = bp2Morgan(tree.interval[0], bps, cMs)
        right = bp2Morgan(tree.interval[1], bps, cMs)
        if right - left >= minLen and (tree.mrca(a,b) != -1 and tree.tmrca(a,b) < maxGen):
            segment_lens.append((right - left, tree.tmrca(a,b)))
    return segment_lens

def multi_run(fun, prms, processes = 4, output=False):
    """Implementation of running in Parallel.
    fun: Function
    prms: The Parameter Files
    processes: How many Processes to use"""
    if output:
        print(f"Running {len(prms)} total jobs; {processes} in parallel.")
    
    if len(prms)>1:
        if output:
            print("Starting Pool of multiple workers...")    
        with mp.Pool(processes = processes) as pool:
            results = pool.starmap(fun, prms)
    elif len(prms)==1:
        if output:
            print("Running single process...")
        results = fun(*prms[0])
    else:
        raise RuntimeWarning("Nothing to run! Please check input.")
    return results

def timeSampling_singlePop_2tp_chrom(gap, Ne, demography=None, chr=20, minLen=4.0, record_full_arg=True):
    # draw a single haplotpye from two time point, one at the present time, and the other $gap generations backward in time
    # return the IBD segments between these two haplotypes
    path2Map = f"/mnt/archgen/users/yilei/Data/Hapmap/genetic_map_GRCh37_chr{chr}.txt"
    recombMap = msprime.RateMap.read_hapmap(path2Map)
    bps, cMs = readHapMap(path2Map)

    # simulate tree sequence
    # why multiple Ne by 2? cuz ploidy is set to one
    if not demography:
        ts = msprime.sim_ancestry(
            samples = [
                msprime.SampleSet(1),
                msprime.SampleSet(1, time=gap)
                ],
            population_size=2*Ne, recombination_rate=recombMap, end_time=1e4,
            record_provenance=False, record_full_arg=record_full_arg, ploidy=1
            )
    else:
        print(f'simulating from a given demography object')
        ts = msprime.sim_ancestry(samples = [msprime.SampleSet(1), msprime.SampleSet(1, time=gap)],
            recombination_rate=recombMap, demography=demography, end_time=1e4,
            record_provenance=False, record_full_arg=record_full_arg, ploidy=1)

    # extract IBD segments
    ibd = ibd_segments_full_ARGs(ts, 0, 1, bps, cMs, maxGen=np.inf, minLen=minLen)
    return ibd

def getPath(tree, s, t):
    # return a list of nodes that specify the path from s to t
    # s is not in the list but t is
    # assume s is lower in the tree than t!
    path = []
    p = tree.parent(s)
    while p != t:
        path.append(p)
        p = tree.parent(p)
    assert(p == t)
    path.append(p)
    return path


def ibd_segments_fullARG_cohort(ts, a, b, bps, cMs, maxGen=np.inf, minLen=4):
    segment_lens = []
    ts = ts.simplify([a,b], keep_unary=True)
    trees_iter = ts.trees()
    tree = next(trees_iter)
    last_mrca = tree.mrca(0, 1)
    last_pathA, last_pathB = getPath(tree, 0, last_mrca), getPath(tree, 1, last_mrca)
    last_left = bp2Morgan(tree.interval[0], bps, cMs)
    segment_lens = []
    for tree in trees_iter:
        mrca = tree.mrca(0, 1)
        pathA, pathB = getPath(tree, 0, mrca), getPath(tree, 1, mrca)
        if mrca != last_mrca  or pathA != last_pathA or pathB != last_pathB:
            left = bp2Morgan(tree.interval[0], bps, cMs)
            if last_mrca <= maxGen and left - last_left >= minLen:
                segment_lens.append(left - last_left)
            last_mrca = mrca
            last_left = left
            last_pathA = pathA
            last_pathB = pathB
    # take care of the last segment
    if last_mrca <= maxGen and cMs[-1] - last_left >= minLen:
        segment_lens.append(cMs[-1] - last_left)
    return segment_lens

def ibd_segments_cohort(ts, a, b, bps, cMs, maxGen=np.inf, minLen=4):
    trees_iter = ts.trees()
    #next(trees_iter) # I don't want the first tree because recombination map doesn't cover it
    tree = next(trees_iter)
    last_mrca = tree.mrca(a, b)
    last_t = tree.time(last_mrca) if last_mrca != -1 else np.inf
    last_left = bp2Morgan(tree.interval[0], bps, cMs)
    segment_lens = []
    for tree in trees_iter:
        mrca = tree.mrca(a, b)
        if mrca != last_mrca:
            left = bp2Morgan(tree.interval[0], bps, cMs)
            if left - last_left >= minLen and last_t < maxGen:
                print(f'{last_left} - {left}')
                segment_lens.append(left - last_left)
            last_mrca = mrca
            last_t = tree.time(mrca) if last_mrca != -1 else np.inf
            last_left = left
    # take care of the last segment
    if last_t <= maxGen and cMs[-1] - last_left >= minLen:
        segment_lens.append(cMs[-1] - last_left)
    return segment_lens

def sort_merge(seglist):
    if len(seglist) == 0:
        return []
    seglist = sorted(seglist, key=lambda seg:seg.left)
    merged = []
    prev_left = seglist[0].left
    prev_right = seglist[0].right
    prev_mrca = seglist[0].node
    for seg in seglist[1:]:
        if seg.node == prev_mrca:
            prev_right = seg.right
        else:
            merged.append((prev_left, prev_right))
            prev_left = seg.left
            prev_right = seg.right
            prev_mrca = seg.node
    merged.append((prev_left, prev_right))
    return merged



def ibd_segments_fast(ts, bkp_bp, bkp_cm, within=None, between=None, max_time=np.inf, minLen=4):
    # return a list of IBD segments that is more recent than maxGen and longer than minLen (given in cM)
    # within and between are as defined in tskit documentation and they are mutually exclusive
    # also, if within is provided, then IBD segments between (0,1), (2,3), (3,4).. etc will be discarded (equivalently, we don't return ROH segments)
    # results4IBDNe is a list of tuples to facilitate producing input for IBDNe
    # tuple: (first sample ID, first sample haplotype index, second sample ID, second sample haplotype index, bp_start, bp_end, length_cM)
    results4IBDNe = []
    if between:
        segs = ts.ibd_segments(between=between, max_time=max_time, store_pairs=True, store_segments=True)
        segment_lens = []
        for pair, _ in segs.items():
            for seg in sort_merge(segs[pair]):
                # determine segment length (in cM)
                bp1, bp2 = seg[0], seg[1]
                i, j = np.searchsorted(bkp_bp, bp1), np.searchsorted(bkp_bp, bp2)
                seglen = bkp_cm[j] - bkp_cm[i]
                if seglen >= minLen:
                    segment_lens.append(seglen)
        return segment_lens
    elif within:
        segs = ts.ibd_segments(within=within, max_time=max_time, store_pairs=True, store_segments=True)
        segment_lens = []
        for pair, _ in segs.items():
            id1, id2 = min(pair[0], pair[1]), max(pair[0], pair[1])
            if id1%2 == 0 and id2%2 == 1 and abs(id1 - id2) == 1:
                continue # ignore ROH
            for seg in sort_merge(segs[pair]):
                # determine segment length (in cM)
                bp1, bp2 = seg[0], seg[1]
                i, j = np.searchsorted(bkp_bp, bp1), np.searchsorted(bkp_bp, bp2)
                seglen = bkp_cm[j] - bkp_cm[i]
                if seglen >= minLen:
                    segment_lens.append(seglen)
                    results4IBDNe.append((f'iid{id1//2}', 1+id1%2, f'iid{id2//2}', 1+id2%2, int(bp1), int(bp2), round(seglen,3)))
        return segment_lens, results4IBDNe
    else:
        raise RuntimeWarning('None of within or between provided. Do nothing...')


def timeSampling_singlePop_2tp_cohort_chrom(gap, Ne, nSample=3, chr=20, minLen=4.0, record_full_arg=True, demography=None, random_seed=1):
    path2Map = f"/mnt/archgen/users/yilei/Data/Hapmap/genetic_map_GRCh37_chr{chr}.txt"
    recombMap = msprime.RateMap.read_hapmap(path2Map)
    bps, cMs = readHapMap(path2Map)

    # simulate tree sequence
    if not demography:
        ts = msprime.sim_ancestry(
            samples = [msprime.SampleSet(nSample), msprime.SampleSet(nSample, time=gap)],
            population_size=Ne, recombination_rate=recombMap,
            record_provenance=False, record_full_arg=record_full_arg, random_seed=random_seed)
    else:
        ts = msprime.sim_ancestry(samples = [msprime.SampleSet(nSample), msprime.SampleSet(nSample, time=gap)],
            recombination_rate=recombMap, demography=demography,
            record_provenance=False, record_full_arg=record_full_arg, random_seed=random_seed)
    print('simulating tree seq done')
    # extract IBD segments
    results = []
    ibd_extractor = ibd_segments_fullARG_cohort if record_full_arg else ibd_segments_cohort
    for id1, id2 in itertools.product(range(2*nSample), range(2*nSample, 4*nSample)):
        results.extend(ibd_extractor(ts, id1, id2, bps, cMs, minLen=minLen))

    return results

def timeSampling_singlePop_2tp_cohort_ind(gap, Ne, nSample=3, chr=range(1,23), minLen=4.0, record_full_arg=True, demography=None, random_seed=1, nprocess=6):
    prms = [[gap, Ne, nSample, chr_, minLen, record_full_arg, demography, random_seed] for chr_ in chr]
    results = multi_run(timeSampling_singlePop_2tp_cohort_chrom, prms, processes=nprocess, output=False)
    aggregated = []
    for result in results:
        aggregated.extend(result)
    aggregated = np.array(aggregated)
    return aggregated

################################ multi time points, recording IBD segments between all pairwise sampling cluster #####################

def timeSampling_singlePop_MultiTP_cohort_chrom(Ne=1e3, nSamples=[], gaps=[], chr=20, minLen=4.0, record_full_arg=True, demography=None, \
            random_seed=1, endTime=1e3, nSamples_model=[], timeInterval=False):
    # Ne: this parameter is ignored if a demography object is given
    path2Map = f"/mnt/archgen/users/yilei/Data/Hapmap/genetic_map_GRCh37_chr{chr}.txt"
    recombMap = msprime.RateMap.read_hapmap(path2Map)
    bps, cMs = readHapMap(path2Map)

    assert(len(gaps)==len(nSamples))
    samples = [msprime.SampleSet(nSample, time=gap) for nSample, gap in zip(nSamples, gaps)]
    # simulate tree sequence
    t1 = time.time()
    if not demography:
        ts = msprime.sim_ancestry(
            samples = samples,
            population_size=Ne, recombination_rate=recombMap,
            record_provenance=False, record_full_arg=record_full_arg, end_time=endTime, random_seed=random_seed)
    else:
        ts = msprime.sim_ancestry(samples = samples,
            recombination_rate=recombMap, demography=demography,
            record_provenance=False, record_full_arg=record_full_arg, end_time=endTime, random_seed=random_seed)
    print(f'simulating tree seq done for ch{chr}, takes {round(time.time() - t1)}s')
    # extract IBD segments
    t1 = time.time()
    results = defaultdict(list) # a hack to make defaultdict picklable

    ############### use tskit's internal IBD extractor, which should be much faster #########################
    bkp_bp = ts.breakpoints(as_array=True)
    bkp_cm = np.array([bp2Morgan(bp, bps, cMs) for bp in bkp_bp])

    nSamples_copy = nSamples if not timeInterval else nSamples_model
    if len(nSamples_copy) == 1:
        ret, results4IBDNe = ibd_segments_fast(ts, bkp_bp, bkp_cm, within=list(range(0, 2*nSamples_copy[0])), max_time=endTime, minLen=minLen)
        results[(0,0)].extend(ret)
        return results, results4IBDNe, chr
    else:
        for i in range(len(nSamples_copy)):
            start_i = 0 if i == 0 else 2*sum(nSamples_copy[:i])
            end_i = 2*sum(nSamples_copy[:i+1])
            for j in range(i, len(nSamples_copy)):
                start_j = 0 if j == 0 else 2*sum(nSamples_copy[:j])
                end_j = 2*sum(nSamples_copy[:j+1])
                if i == j:
                    ret, _ = ibd_segments_fast(ts, bkp_bp, bkp_cm, within=list(range(start_i, end_i)), max_time=endTime, minLen=minLen)
                    results[(i,j)].extend(ret)
                else:
                    results[(i,j)].extend(ibd_segments_fast(ts, bkp_bp, bkp_cm, \
                        between=[list(range(start_i, end_i)), list(range(start_j, end_j))], \
                            max_time=endTime, minLen=minLen))
        print(f'extracting IBD done for ch{chr}, takes {round(time.time() - t1)}s')
        return results, chr


    ibd_extractor = ibd_segments_fullARG_cohort if record_full_arg else ibd_segments_cohort
    for i in range(len(nSamples)):
        start_i = 0 if i == 0 else 2*sum(nSamples[:i])
        end_i = 2*sum(nSamples[:i+1])
        for j in range(i, len(nSamples)):
            start_j = 0 if j == 0 else 2*sum(nSamples[:j])
            end_j = 2*sum(nSamples[:j+1])
            for id1, id2 in itertools.product(range(start_i, end_i), range(start_j, end_j)):
                if i == j and id2 <= 1 + id1: # don't count ROH and don't double count when recording IBD within the same sampling cluster
                    continue
                results[(i, j)].extend(ibd_extractor(ts, id1, id2, bps, cMs, maxGen=endTime, minLen=minLen))
    print(f'extracting IBD done for ch{chr}, takes {round(time.time() - t1)}s')
    return results, chr

def timeSampling_singlePop_MultiTP_cohort_ind(Ne=1e3, nSamples=[], gaps=[], chr=range(1,23), minLen=4.0, record_full_arg=True, \
        demography=None, random_seed=1, endTime=1e3, nprocess=6, nSamples_model=[], timeInterval=False):
    prms = [[Ne, nSamples, gaps, chr_, minLen, record_full_arg, demography, random_seed, endTime, nSamples_model, timeInterval] for chr_ in chr]
    results = multi_run(timeSampling_singlePop_MultiTP_cohort_chrom, prms, processes=nprocess, output=False)
    aggregated = defaultdict(dict)
    numSampleCluster = len(nSamples) if not timeInterval else len(nSamples_model)
    if numSampleCluster > 1:
        for result, chr in results:
            print(f'processing segments from ch{chr}')
            for k, v in result.items():
                aggregated[k][chr] = v # store segments from different chromosomes separately, for bootstraping later
        return aggregated
    else:
        IBDNe = {}
        for result, results4IBDNe, chr in results:
            IBDNe[chr] = results4IBDNe
            print(f'processing segments from ch{chr}')
            for k, v in result.items():
                aggregated[k][chr] = v # store segments from different chromosomes separately, for bootstraping later
        return aggregated, IBDNe

def maskIBD(path2unmaskedIBD, path2ChrDelimiter, path2mask):
    """
    Given a file containing a list of (unmasked) IBD (in the format of Harald's ancIBD output)
    write a tsv file containing a list of masked IBD, in the following format:
    iid1, iid2, ch, startCM, endCM, lengthCM
    Note that the ch index refers to the "artificial" chromosome created after applyingi the mask.
    This function is adapted from relevant part of the function analytic_multi.inferVecNe_singlePop_MultiTP_withMask()
    """
    chrDelimiter = {}
    with open(path2ChrDelimiter) as f:
        for line in f:
            chr, start, end = line.strip().split()
            start, end = float(start), float(end)
            start = max(0.0, start)
            chrDelimiter[int(chr)] = (float(start), float(end))


    masks = defaultdict(lambda: [])
    with open(path2mask) as f:
        for line in f:
            chr, bp_start, bp_end, cm_start, cm_end = line.strip().split()
            masks[int(chr)].append((float(cm_start), float(cm_end)))

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

    # print(f'After masking, {len(chr_span_dict)} artificial chromosomes are created from {len(chrDelimiter)} original chromosomes.')
    # ch_len_dict = {k: v[2]-v[1] for k, v in chr_span_dict.items()}
    spanlist = [(v[1], v[2]) for k, v in chr_span_dict.items()]
    
    #### Now mask IBD segments and assign them to the correct artificial chromosomes
    with open('masked.IBD.tsv', 'w') as out:
        with open(path2unmaskedIBD) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                _, _, StartM, EndM, _, lengthM, ch, iid1, iid2, *_ = line.strip().split()
                ch, ibd_start, ibd_end, length = int(ch), 100*float(StartM), 100*float(EndM), 100*float(lengthM)
                for i, span in enumerate(spanlist[map[ch]:map[ch+1]]):
                    # ask whether the current segment being read has any overlap with this span of interest
                    span_start, span_end = span
                    if ibd_end <= span_start or ibd_start >= span_end:
                        continue
                    else:
                        ibd_start_masked, ibd_end_masked = max(span_start, ibd_start), min(span_end, ibd_end)
                        out.write(f'{iid1}\t{iid2}\t{i+map[ch]}\t{ibd_start_masked}\t{ibd_end_masked}\t{ibd_end_masked-ibd_start_masked}\n')



############################################# simulations with IM model ##############################################################
######################################################################################################################################
######################################################################################################################################

def timeSampling_IM_2TP_cohort_chrom(demography, nSamples, timeDiff, chr=20, minLen=4.0, record_full_arg=False, panmictic=False, random_seed=1, endTime=1e3):
    # Ne: this parameter is ignored if a demography object is given
    path2Map = f"/mnt/archgen/users/yilei/Data/Hapmap/genetic_map_GRCh37_chr{chr}.txt"
    recombMap = msprime.RateMap.read_hapmap(path2Map)
    bps, cMs = readHapMap(path2Map)

    if not panmictic:
        samplesA = [msprime.SampleSet(nSamples, population='A', time=0)]
        samplesB = [msprime.SampleSet(nSamples, population='B', time=timeDiff)]
        samples = samplesA + samplesB
    else:
        samplesA = [msprime.SampleSet(nSamples, time=0)]
        samplesB = [msprime.SampleSet(nSamples, time=timeDiff)]
        samples = samplesA + samplesB

    # simulate tree sequence
    t1 = time.time()
    ts = msprime.sim_ancestry(samples = samples,
        recombination_rate=recombMap, demography=demography,
        record_provenance=False, record_full_arg=record_full_arg, end_time=endTime, random_seed=random_seed)
    print(f'simulating tree seq done for ch{chr}, takes {round(time.time() - t1)}s')
    # extract IBD segments
    t1 = time.time()
    results = defaultdict(list) # a hack to make defaultdict picklable

    ############### use tskit's internal IBD extractor, which should be much faster #########################
    bkp_bp = ts.breakpoints(as_array=True)
    bkp_cm = np.array([bp2Morgan(bp, bps, cMs) for bp in bkp_bp])

    start_i = 0
    end_i = 2*nSamples
    start_j = 2*nSamples
    end_j = 4*nSamples
    print(f'find IBD between ({start_i}-{end_i}) and ({start_j}-{end_j})')
    results[(0,1)].extend(ibd_segments_fast(ts, bkp_bp, bkp_cm, \
            between=[list(range(start_i, end_i)), list(range(start_j, end_j))], \
            max_time=endTime, minLen=minLen))

    print(f'find IBD within ({start_i}-{end_i})')
    ret, _ = ibd_segments_fast(ts, bkp_bp, bkp_cm, \
            within=list(range(start_i, end_i)), max_time=endTime, minLen=minLen)
    results[(0,0)].extend(ret)

    print(f'find IBD within ({start_j}-{end_j})')
    ret, _ = ibd_segments_fast(ts, bkp_bp, bkp_cm, \
            within=list(range(start_j, end_j)), max_time=endTime, minLen=minLen)
    results[(1,1)].extend(ret)

    print(f'extracting IBD done for ch{chr}, takes {round(time.time() - t1)}s')
    return results, chr

def timeSampling_IM_2TP_cohort_ind(demography, nSamples, timeDiff, chr=range(1,23), minLen=4.0, record_full_arg=False, panmictic=False, random_seed=1, endTime=1e3, nprocess=6):
    prms = [[demography, nSamples, timeDiff, chr_, minLen, record_full_arg, panmictic, random_seed, endTime] for chr_ in chr]
    results = multi_run(timeSampling_IM_2TP_cohort_chrom, prms, processes=nprocess, output=False)
    aggregated = defaultdict(dict)
    for result, chr in results:
        print(f'processing segments from ch{chr}')
        for k, v in result.items():
            aggregated[k][chr] = v # store segments from different chromosomes separately, for bootstraping later
    return aggregated

############################################# end of simulations with IM model #######################################################
######################################################################################################################################
######################################################################################################################################

def timeSampling_IM_2TP_cohort_chrom_timeRange(demography, countNumSampleAtEachTime1, countNumSampleAtEachTime2, chr=20, minLen=4.0, record_full_arg=False, panmictic=False, random_seed=1, endTime=1e3):
    # Ne: this parameter is ignored if a demography object is given
    path2Map = f"/mnt/archgen/users/yilei/Data/Hapmap/genetic_map_GRCh37_chr{chr}.txt"
    recombMap = msprime.RateMap.read_hapmap(path2Map)
    bps, cMs = readHapMap(path2Map)

    samples = []
    if not panmictic:
        for sampleTime, nSample in countNumSampleAtEachTime1.items():
            samples += [msprime.SampleSet(nSample, population='A', time=sampleTime)]
        for sampleTime, nSample in countNumSampleAtEachTime2.items():
            samples += [msprime.SampleSet(nSample, population='B', time=sampleTime)]
    else:
        for sampleTime, nSample in countNumSampleAtEachTime1.items():
            samples += [msprime.SampleSet(nSample, time=sampleTime)]
        for sampleTime, nSample in countNumSampleAtEachTime2.items():
            samples += [msprime.SampleSet(nSample, time=sampleTime)]

    # simulate tree sequence
    t1 = time.time()
    ts = msprime.sim_ancestry(samples = samples,
        recombination_rate=recombMap, demography=demography,
        record_provenance=False, record_full_arg=record_full_arg, end_time=endTime, random_seed=random_seed)
    print(f'simulating tree seq done for ch{chr}, takes {round(time.time() - t1)}s')
    # extract IBD segments
    t1 = time.time()
    results = defaultdict(list) # a hack to make defaultdict picklable

    ############### use tskit's internal IBD extractor, which should be much faster #########################
    bkp_bp = ts.breakpoints(as_array=True)
    bkp_cm = np.array([bp2Morgan(bp, bps, cMs) for bp in bkp_bp])

    nSample1 = sum(n for _, n in countNumSampleAtEachTime1.items())
    nSample2 = sum(n for _, n in countNumSampleAtEachTime2.items())
    start_i = 0
    end_i = 2*nSample1
    start_j = 2*nSample1
    end_j = 2*(nSample1 + nSample2)
    print(f'find IBD between ({start_i}-{end_i}) and ({start_j}-{end_j})')
    results[(0,1)].extend(ibd_segments_fast(ts, bkp_bp, bkp_cm, \
            between=[list(range(start_i, end_i)), list(range(start_j, end_j))], \
            max_time=endTime, minLen=minLen))

    print(f'find IBD within ({start_i}-{end_i})')
    ret, _ = ibd_segments_fast(ts, bkp_bp, bkp_cm, \
            within=list(range(start_i, end_i)), max_time=endTime, minLen=minLen)
    results[(0,0)].extend(ret)

    print(f'find IBD within ({start_j}-{end_j})')
    ret, _ = ibd_segments_fast(ts, bkp_bp, bkp_cm, \
            within=list(range(start_j, end_j)), max_time=endTime, minLen=minLen)
    results[(1,1)].extend(ret)

    print(f'extracting IBD done for ch{chr}, takes {round(time.time() - t1)}s')
    return results, chr



def timeSampling_IM_2TP_cohort_ind_timeRange(demography, nSamples, timeDiff, radius=0, chr=range(1,23), minLen=4.0, record_full_arg=False, panmictic=False, random_seed=1, endTime=1e3, nprocess=6):
    time1 = radius
    time2 = radius + timeDiff
    timeRange1 = np.arange(0, 2*radius+1)
    timeRange2 = np.arange(time2-radius, time2+radius+1)
    sampledTime1 = random.choices(timeRange1, k=nSamples)
    countNumSampleAtEachTime1 = Counter(sampledTime1) # a dictionary in the format of time:nSample for population A
    sampledTime2 = random.choices(timeRange2, k=nSamples)
    countNumSampleAtEachTime2 = Counter(sampledTime2) # a dictionary in the format of time:nSample for population B
    print(countNumSampleAtEachTime1)
    print(countNumSampleAtEachTime2)
    
    prms = [[demography, countNumSampleAtEachTime1, countNumSampleAtEachTime2, chr_, minLen, record_full_arg, panmictic, random_seed, endTime] for chr_ in chr]
    results = multi_run(timeSampling_IM_2TP_cohort_chrom_timeRange, prms, processes=nprocess, output=False)
    aggregated = defaultdict(dict)
    for result, chr in results:
        print(f'processing segments from ch{chr}')
        for k, v in result.items():
            aggregated[k][chr] = v # store segments from different chromosomes separately, for bootstraping later
    return aggregated