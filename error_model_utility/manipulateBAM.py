import argparse
import os
import subprocess
import sys
import numpy as np
from scipy.interpolate import interp1d

"""
This script manipulates BAM files to simulate IBD segments on one chromosome between two diploid samples.
It takes in 4 bam files and output 2 bam files.
--bam1 is the BAM file of a mother.
--bam2 is the BAM file of a son.
--bam3 is the BAM file of a female sample.
--bam4 is the BAM file of another female sample.
bam1 and bam3 are mixed to form a synthetic diploid chromosome.
bam2 and bam4 are mixed to form another synthetic diploid chromosome.
The mother-son pair forms a natural IBD segment.
Additionally, it needs a genetic map for the chromosome you wish to simulate. In the same directory we provide a genetic map for human chromosome 3.
If you wish to simulate a different chromosome, you can replace this genetic map with a map of your choice.
Finally, --c1, --c2, --c3, --c4 are the coverage of the 4 bam files on the chromosome you wish to simulate.
-t is the coverage you wish to simulate for the two synthetic diploid chromosomes.
-l is the length of the IBD segment you wish to simulate.
--seed is the random seed, and it controls where the simulated IBD segment is placed on the chromosome.
--dryrun is a flag that allows you to see the start and end positions of the IBD segment without actually mixing the BAM files.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam1", type=str, help="bam file1")
    parser.add_argument("--bam2", type=str, help="bam file2")
    parser.add_argument("--bam3", type=str, help="bam file3")
    parser.add_argument("--bam4", type=str, help="bam file4")
    parser.add_argument("--c1", type=float, help="coverage of bam1")
    parser.add_argument("--c2", type=float, help="coverage of bam2")
    parser.add_argument("--c3", type=float, help="coverage of bam3")
    parser.add_argument("--c4", type=float, help="coverage of bam4")
    parser.add_argument("--map", type=str, 
            default='/mnt/archgen/users/yilei/bin/ancIBD_data/afs/v51.1_1240k.chr3.map', 
            help="genetic map")
    parser.add_argument("-t", type=float, help="target coverage")
    parser.add_argument("-l", type=float, help="length of IBD to simulate")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--dryrun", action='store_true', help="output bam file")
    args = parser.parse_args()

    # read in genetic map
    bps = []
    cms = []
    with open(args.map, "r") as f:
        for line in f:
            rsID, ch, Morgan, bp, ref, alt = line.strip().split()
            bps.append(int(bp))
            cms.append(100*float(Morgan))
    bps = np.array(bps)
    cms = np.array(cms)
    # create a interpolator from cms to bps
    cms_to_bps_interp1d = interp1d(cms, bps, kind="linear")
    bps_to_cms_interp1d = interp1d(bps, cms, kind="linear")

    # determine the start and end BP positions of the IBD to simulate
    # take a uniform draw between cm_start and cm_end - args.l
    # and then add args.l to get the end point
    # set random seed for numpy
    np.random.seed(args.seed)
    cm_IBD_start = np.random.uniform(cms[0], cms[-1] - args.l)
    cm_IBD_end = cm_IBD_start + int(args.l)
    bp_IBD_start = int(cms_to_bps_interp1d(cm_IBD_start))
    bp_IBD_end = int(cms_to_bps_interp1d(cm_IBD_end))
    print("IBD start: ", bp_IBD_start, round(cm_IBD_start, 3),  "IBD end: ", bp_IBD_end, round(cm_IBD_end, 3), flush=True)
    if args.dryrun:
        sys.exit(0)

    ######################################################################################################################################
    frac1 = args.t/args.c1
    frac2 = args.t/args.c2
    frac3 = args.t/args.c3
    frac4 = args.t/args.c4
    print('subsampling fraction', frac1, frac2, frac3, frac4, flush=True)
    # use numpy to generate a random integer between 1 and 10000000
    subsample_seed = np.random.randint(1, int(1e7) + 1)
    cmd = f'samtools view --subsample {frac3} -b --subsample-seed {subsample_seed} -o tmp1.bam {args.bam3} 3:1-{bp_IBD_start-1} 3:{bp_IBD_end+1}'
    os.system(cmd)
    os.system('samtools index tmp1.bam')
    # downsample the BAM file of the mother 
    # extract reads in the IBD region from the BAM file of the mother
    cmd = f'samtools view -b --subsample {frac1} -b --subsample-seed {subsample_seed+1} -o tmp2.bam {args.bam1} 3:{bp_IBD_start}-{bp_IBD_end}'
    os.system(cmd)
    os.system('samtools index tmp2.bam')
    # merge the two BAM files
    os.system('samtools merge -o ind1.bam tmp1.bam tmp2.bam')
    os.system('samtools index ind1.bam')
    ### remove temporary files
    os.system('rm tmp1.bam tmp1.bam.bai tmp2.bam tmp2.bam.bai')

    # downsample the BAM file of the third individual
    subprocess.run(f'samtools view --subsample {frac2} -b --subsample-seed {subsample_seed+2} -o tmp1.bam {args.bam2} 3:{bp_IBD_start}-{bp_IBD_end}', shell=True)
    os.system('samtools index tmp1.bam')
    subprocess.run(f'samtools view --subsample {frac4} -b --subsample-seed {subsample_seed+3} -o tmp2.bam {args.bam4} 3:1-{bp_IBD_start-1} 3:{bp_IBD_end+1}', shell=True)
    os.system('samtools index tmp2.bam')
    os.system('samtools merge -o ind2.bam tmp1.bam tmp2.bam')
    os.system('samtools index ind2.bam')
    ### remove temporary files
    os.system('rm tmp1.bam tmp1.bam.bai tmp2.bam tmp2.bam.bai')
