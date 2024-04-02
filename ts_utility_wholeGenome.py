import msprime
import numpy as np
import time
from collections import defaultdict
from ts_utility import sort_merge
import os
import sys
import tskit
## simulate the entire 22 chromosomes together (instead of independently)
## for simplicity, we assume uniform recombination rate

import math

###################################### construct a uniform recombination rate map ######################################
r_chrom = 1e-8
r_break = math.log(2)
chrom_lengths = np.array([286279234, 268839622, 223361095, 214688476, 204089357, 192039918,
       187220500, 168003442, 166359329, 181144008, 158218650, 174679023,
       125706316, 120202583, 141860238, 134037725, 128490529, 117708923,
       107733846, 108266934,  62786478,  74109562])
chrom_positions = np.insert(np.cumsum(chrom_lengths), 0, 0)
map_positions_uniform = [chrom_positions[0], chrom_positions[1]]
for i in range(1, len(chrom_positions)-1):
    map_positions_uniform.append(chrom_positions[i] + 1)
    map_positions_uniform.append(chrom_positions[i+1])

rates = [r_chrom, r_break] * (len(chrom_lengths) - 1) + [r_chrom]
# print(f'length of map_positions: {len(map_positions_uniform)}')
# print(f'length of rates: {len(rates)}')
rate_map = msprime.RateMap(position=map_positions_uniform, rate=rates)

###################################### construct a variable recombination rate map (use Amy's simplified recomb map from Hapmap) ######################################
ch2map = {}
for ch in np.arange(1,23):
    ratemap = msprime.RateMap.read_hapmap(f'/mnt/archgen/users/yilei/Data/Hapmap/genetic_map_GRCh37_chr{ch}.txt')
    ch2map[ch] = ratemap
chrom_lengths_hapmap = np.array([ch2map[ch].right[-1] for ch in np.arange(1,23)]).astype(int)
#chrom_positions_hapmap = np.insert(np.cumsum(chrom_lengths_hapmap), 0, 0)
chrom_positions_hapmap = [0]
map_positions_hapmap = []
rates_hapmap = []
map_positions_hapmap.extend(ch2map[1].left)
map_positions_hapmap.append(ch2map[1].right[-1])
chrom_positions_hapmap.append(ch2map[1].right[-1])
rates_hapmap.extend(ch2map[1].rate)
for ch in np.arange(2,23):
    map_positions_hapmap.append(chrom_positions_hapmap[-1]+1)
    rates_hapmap.append(np.log(2))
    map_positions_hapmap.extend(chrom_positions_hapmap[-1] + ch2map[ch].left[1:])
    map_positions_hapmap.append(chrom_positions_hapmap[-1] + ch2map[ch].right[-1])
    chrom_positions_hapmap.append(chrom_positions_hapmap[-1] + ch2map[ch].right[-1])
    rates_hapmap.extend(ch2map[ch].rate)
rate_map_hapmap = msprime.RateMap(position=map_positions_hapmap, rate=rates_hapmap)
##################################################################################################################################################3



nSNPs = {1:88409, 2:93876, 3:77346, 4:68519, 5:69064, 6:75348, 7:59604, 8:60829, 9:50547, \
         10:58611, 11:54591, 12:53738, 13:38928, 14:35886, 15:34281, 16:34336, 17:28893, 18:33847, \
            19:18093, 20:28941, 21:15708, 22:15484}

# def splitSegmentOverChrom_uniformMap(bp1, bp2):
#     # first, determine if the given segment spans across chromosomes
#     # if so, split it into multiple segments
#     # return a list of tuples of (chromosome id, segment lengths (in cM)) as a result of this splitting
#     # if not, return a list containing the length of the given segment
#     chrom1, chrom2 = np.searchsorted(chrom_positions, [bp1, bp2])
#     if chrom1 == chrom2:
#         return [(chrom1, (bp2-bp1)/1e6)]
#     else:
#         seglens = []
#         seglens.append((chrom1, (chrom_positions[chrom1]-bp1)/1e6))
#         for i in range(chrom1+1, chrom2):
#             seglens.append((i, chrom_lengths[i]/1e6))
#         seglens.append((chrom2, (bp2-chrom_positions[chrom2-1])/1e6))
#         return seglens

# def splitSegmentOverChrom_hapmap(bp1, bp2):
#     chrom1, chrom2 = np.searchsorted(chrom_positions_hapmap, [bp1, bp2])
#     if chrom_positions_hapmap[chrom1] == bp1:
#         chrom1 += 1
#     assert(chrom1 >= 1 and chrom1 <= 22)
#     assert(chrom2 >= 1 and chrom2 <= 22)
#     bp1 -= chrom_positions_hapmap[chrom1-1]
#     bp2 -= chrom_positions_hapmap[chrom2-1]
#     if chrom1 == chrom2:
#         m1, m2 = ch2map[chrom1].get_cumulative_mass(bp1), ch2map[chrom2].get_cumulative_mass(bp2)
#         return [(chrom1, (m2 - m1)*100)]
#     else:
#         seglens = []
#         seglens.append((chrom1, (ch2map[chrom1].total_mass - ch2map[chrom1].get_cumulative_mass(bp1))*100))
#         for i in range(chrom1+1, chrom2):
#             seglens.append((i, 100*ch2map[i].total_mass))
#         seglens.append((chrom2, 100*ch2map[chrom2].get_cumulative_mass(bp2)))
#         return seglens
    
def splitSegmentOverChrom(bp1, bp2, bp2chAndCM):
    ch1, cM1 = bp2chAndCM[bp1]
    ch2, cM2 = bp2chAndCM[bp2]
    if ch1 == ch2:
        return [(ch1, cM2-cM1)]
    else:
        seglens = []
        seglens.append((ch1, 100*ch2map[ch1].total_mass - cM1))
        for i in range(ch1+1, ch2):
            seglens.append((i, 100*ch2map[i].total_mass))
        seglens.append((ch2, cM2))
        return seglens


def ibd_segments(ts, bp2chAndCM, within=None, between=None, max_time=2500, minLen=4, uniform_map=True):
    # return a list of IBD segments that is more recent than maxGen and longer than minLen (given in cM)
    # within and between are as defined in tskit documentation and they are mutually exclusive
    # also, if within is provided, then IBD segments between (0,1), (2,3), (3,4).. etc will be discarded (equivalently, we don't return ROH segments)
    # results4IBDNe is a list of tuples to facilitate producing input for IBDNe
    # tuple: (first sample ID, first sample haplotype index, second sample ID, second sample haplotype index, bp_start, bp_end, length_cM)
    lst = lambda:defaultdict(list)
    pair2ibd = defaultdict(lst)
    results4IBDNe = []
    if between:
        segs = ts.ibd_segments(between=between, max_time=max_time, store_pairs=True, store_segments=True)
        for pair, _ in segs.items():
            for seg in sort_merge(segs[pair]):
                # determine segment length (in cM)
                bp1, bp2 = seg[0], seg[1]
                # determine whether this segment "spans across" chromosomes
                seglens = splitSegmentOverChrom(bp1, bp2, bp2chAndCM)
                for tuple in seglens:
                    ch, seglen = tuple
                    if seglen >= minLen:
                        pair2ibd[pair][ch].append(seglen)
        return pair2ibd
    elif within:
        segs = ts.ibd_segments(within=within, max_time=max_time, store_pairs=True, store_segments=True)
        chromDelimiter = chrom_positions if uniform_map else np.array(chrom_positions_hapmap)
        for pair, _ in segs.items():
            id1, id2 = min(pair[0], pair[1]), max(pair[0], pair[1])
            if id1%2 == 0 and id2%2 == 1 and abs(id1 - id2) == 1:
                continue # ignore ROH
            for seg in sort_merge(segs[pair]):
                # determine segment length (in cM)
                bp1, bp2 = seg[0], seg[1]
                seglens = splitSegmentOverChrom(bp1, bp2, bp2chAndCM)
                for tuple in seglens:
                    ch, seglen = tuple
                    if seglen >= minLen:
                        pair2ibd[pair][ch].append(seglen)
                        bp1 -= chromDelimiter[ch-1]
                        bp2 -= chromDelimiter[ch-1]
                        results4IBDNe.append((f'iid{id1//2}', 1+id1%2, f'iid{id2//2}', 1+id2%2, ch, int(bp1), int(bp2), round(seglen,3)))
        return pair2ibd, results4IBDNe
    else:
        raise RuntimeWarning('None of within or between provided. Do nothing...')
    



def simulate_wholeGenome(demography, nSamples=[], gaps=[], minLen=4.0, record_full_arg=False, \
            random_seed=1, endTime=None, nSamples_model=[], timeInterval=False, population=None, tsSavePath=None, uniform_map=True):
    assert(len(gaps)==len(nSamples))
    if population is None:
        samples = [msprime.SampleSet(nSample, time=gap) for nSample, gap in zip(nSamples, gaps)]
    else:
        samples = [msprime.SampleSet(nSample, time=gap, population=population) for nSample, gap in zip(nSamples, gaps)]
    # simulate tree sequence
    if not os.path.exists(tsSavePath):
        t1 = time.time()
        rateMap = rate_map if uniform_map else rate_map_hapmap
        ts = msprime.sim_ancestry(samples = samples,
            recombination_rate=rateMap, demography=demography,
            record_provenance=False, record_full_arg=record_full_arg, end_time=endTime, \
            model=[msprime.DiscreteTimeWrightFisher(duration=500), msprime.StandardCoalescent(),], random_seed=random_seed)

        print(f'simulating tree seq done, takes {round(time.time() - t1)}s')
        if tsSavePath:
            ts.dump(tsSavePath)
    else:
        print(f'tree sequence already exists, load from {tsSavePath}')
    
    ### now extract IBD segments
    t1 = time.time()
    ts = tskit.load(tsSavePath)

    ########## if hapmap is used, save bp->cm mapping for later use, to repetitive computation time ##########
    bp2chAndCM = {}
    breakpointsbp = ts.breakpoints(as_array=True)
    for bp in breakpointsbp:
        chrom = np.searchsorted(chrom_positions_hapmap, bp)
        if bp == 0:
            chrom = 1
        # if chrom_positions_hapmap[chrom] == bp:
        #     chrom += 1
        if not (chrom >= 1 and chrom <= 22):
            print(f'chrom {chrom} not in range 1-22, bp={bp}')
            sys.exit()
        assert(chrom >= 1 and chrom <= 22)
        bp_ = bp
        if chrom > 1:
            bp_ -= chrom_positions_hapmap[chrom-1]
        cM = 100*(ch2map[chrom].get_cumulative_mass(bp_))
        bp2chAndCM[bp] = (chrom, cM)

    nSamples_copy = nSamples if not timeInterval else nSamples_model
    # results map contains information of IBD segments for each pair of sampled haplotypes
    # it is of the form (id1, id2):[a list of ibd segments]
    # set the max_time for extracting IBD semgents to 2500 to avoid excessive memory usage
    results = {}
    print('extracting IBD segments...')
    if len(nSamples_copy) == 1:
        ret, results4IBDNe = ibd_segments(ts, bp2chAndCM, within=list(range(0, 2*nSamples_copy[0])), max_time=2500, minLen=minLen, uniform_map=uniform_map)
        results.update(ret)
        print(f'extracting IBD done, takes {round(time.time() - t1)}s')
        return results, results4IBDNe
    else:
        for i in range(len(nSamples_copy)):
            start_i = 0 if i == 0 else 2*sum(nSamples_copy[:i])
            end_i = 2*sum(nSamples_copy[:i+1])
            for j in range(i, len(nSamples_copy)):
                start_j = 0 if j == 0 else 2*sum(nSamples_copy[:j])
                end_j = 2*sum(nSamples_copy[:j+1])
                if i == j:
                    ret, _ = ibd_segments(ts, bp2chAndCM, within=list(range(start_i, end_i)), max_time=2500, minLen=minLen, uniform_map=uniform_map)
                    results.update(ret)
                else:
                    results.update(ibd_segments(ts, bp2chAndCM, \
                        between=[list(range(start_i, end_i)), list(range(start_j, end_j))], \
                            max_time=2500, minLen=minLen, uniform_map=uniform_map))
        print(f'extracting IBD done, takes {round(time.time() - t1)}s')
        return results
    
def write2vcf_wholeGenome(ts, outFolder, random_seed=1, mutation_rate=1.25e-8, uniform_map=True):
    chromDelimiter = chrom_positions if uniform_map else np.array(chrom_positions_hapmap)
    print(f'chromDelimiter: {chromDelimiter}')
    nSamples = ts.num_samples # this gives haploid sample size
    mts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=random_seed)

    for ch in range(1, 23):
        t1 = time.time()
        mts_ch = mts.keep_intervals(np.array([1+chromDelimiter[ch-1], chromDelimiter[ch]]).reshape(1,2))
        site2keep = np.zeros(mts_ch.num_sites)
        for variant in mts_ch.variants():
            site2keep[variant.site.id] = np.sum(variant.genotypes)/(nSamples) > 0.05 and np.sum(variant.genotypes)/(nSamples) < 0.95
        sitemask = np.full(mts_ch.num_sites, True)
        nsnps = int(min(nSNPs[ch], np.sum(site2keep)))
        if nsnps < nSNPs[ch]:
            print(f'chr{ch} has less than {nSNPs[ch]} SNPs, only {nsnps} SNPs are kept.')
        sitemask[np.random.choice(np.where(site2keep==True)[0], size=nsnps, replace=False)] = False
        if os.path.isfile(os.path.join(outFolder, f"chr{ch}.vcf.gz")):
            os.remove(os.path.join(outFolder, f"chr{ch}.vcf.gz"))
        lst = lambda x : x - chromDelimiter[ch-1]
        mts_ch.write_vcf(open(os.path.join(outFolder, f"chr{ch}.vcf"), 'w'), site_mask=sitemask, contig_id=ch, position_transform=lst)
        os.system('bgzip ' + os.path.join(outFolder, f"chr{ch}.vcf"))
        print(f"writing vcf for chr{ch} done, takes {round(time.time()-t1, 3)} seconds.", flush=True)
