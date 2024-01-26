import msprime
import numpy as np
from ..ts_utility import readHapMap, ibd_segments_full_ARGs, ibd_segments_full_ARGs_plusTMRCA

def IM_chrom(demography, chr=20, minLen=4.0, record_full_arg=True, samplingtime=(0,0), record_tmrca=False):
    # draw a single haplotpye from two time point, one at the present time, and the other $gap generations backward in time
    # return the IBD segments between these two haplotypes
    path2Map = f"/mnt/archgen/users/yilei/Data/Hapmap/genetic_map_GRCh37_chr{chr}.txt"
    recombMap = msprime.RateMap.read_hapmap(path2Map)
    bps, cMs = readHapMap(path2Map)

    # simulate tree sequence
    ts = msprime.sim_ancestry(samples = [
        msprime.SampleSet(1, population='A', time=samplingtime[0], ploidy=1),
        msprime.SampleSet(1, population='B', time=samplingtime[1], ploidy=1)],
        recombination_rate=recombMap, demography=demography, end_time=1e3,
        record_provenance=False, record_full_arg=record_full_arg)

    # extract IBD segments
    if not record_tmrca:
        ibd = ibd_segments_full_ARGs(ts, 0, 1, bps, cMs, maxGen=np.inf, minLen=minLen)
    else:
        ibd = ibd_segments_full_ARGs_plusTMRCA(ts, 0, 1, bps, cMs, maxGen=np.inf, minLen=minLen)
    return ibd

