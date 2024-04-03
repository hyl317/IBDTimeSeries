import argparse
import os
import pickle
import numpy as np
from analytic_multi import inferVecNe_singlePop_MultiTP_withMask

def main():
    parser = argparse.ArgumentParser(description='Convert bam file to hdf5 format that stores readcount info at target sites.')
    parser.add_argument('--IBD', action="store", dest="path2IBD", type=str, required=True,
                        help="A .tsv file containig the IBD segments. It should at least have columns named iid1, iid2, ch, StartM, EndM, lengthM.")
    parser.add_argument('--date', action="store", dest="path2SampleAge", type=str, required=True,
                        help="A tab-separated txt file containing the sample age information. One sample per line. \
                        The first column should be sample id and the second column should be sample date in BP.")
    parser.add_argument('--chrdelim', action="store", dest="chrdelim", type=str, required=True,
                        help='A tab-separated txt file containing the genetic map position (in cM) of the first and last SNP on each chromosome.\
                            One line per chromosome. The first column is the chromosome id, the second column is the genetic map position of the first SNP, \
                            and the third column is the genetic map position of the last SNP. \
                            This is used to determine the total genetic map length \
                            (and to intersect with the mask file, if provided, to determine the total unmasked map length).')
    parser.add_argument('--mask', action="store", dest="path2mask", type=str, required=False, default=None,
                        help='A tab-separated txt file containing the mask information. \
                            One line per masked region. The first column is the chromosome id.\
                            The second, third columns are the start and end physical position (in bp) of the masked region.\
                            The fourth, fifth columns are the start and end genetic map position (in cM) of the masked region.') 
    parser.add_argument('--Tmax', action='store', dest='Tmax', type=int, required=False, default=100, 
                        help='The maximum number of generations back to infer Ne. Default is 100.')
    parser.add_argument('--minN', action='store', dest='minN', type=int, required=False, default=10, 
                        help='The minimum number of samples in a sample point. Default is 10. Sample points with fewer samples will be discarded.')
    parser.add_argument('--minl_calc', action='store', dest='minl_calc', type=float, required=False, default=2.0, 
                        help='The minimum length of IBD segments for which the error model will be applied. Default is 2.0 cM.')
    parser.add_argument('--minl_infer', action='store', dest='minl_infer', type=float, required=False, default=8.0,
                        help='The minimum length of IBD segments to be used for Ne inference. Default is 8.0 cM.')
    parser.add_argument('--maxl_calc', action='store', dest='maxl_calc', type=float, required=False, default=25.0, 
                        help='The maximum length of IBD segments for which the error model will be applied. Default is 25.0 cM.')
    parser.add_argument('--maxl_infer', action='store', dest='maxl_infer', type=float, required=False, default=20.0,
                        help='The maximum length of IBD segments to be used for Ne inference. Default is 20.0 cM.')
    parser.add_argument('--merge', action='store', dest='merge', type=int, required=False, default=5, 
                        help='Merge all samples within the --merge generations to one single time point. Default is 5.\
                            This is calculated as follows. The time is counted from the oldest sample, \
                            and all other samples will be assigned to time intervals defined as np.arang(eoldest_sample_time, youngest_sample_time, merge*generation_time).')
    parser.add_argument('--generation_time', action='store', dest='generation_time', type=float, required=False, default=29,
                        help='The generation time in years. Default is 29.')
    parser.add_argument('--alpha', action='store', dest='alpha', type=float, required=False, default=10000, 
                        help='Set the regularization parameter for the l2 penalty. Default is 1e4. \
                        Note that if --autocv is set to True, this value will be overriden. If you want to a specific value for alpha, do not set the --autocv flag')
    parser.add_argument('--fp', action='store', dest='fp', type=str, required=False, default='', 
                        help='A pickled python object for computing false positive rate at various segment length.\
                        We use scipy.interpolate.interp1d, but any pickled callable object that takes exactly one argument (segment length) and returns the false positive rate at that length will work.')
    parser.add_argument('--recall', action='store', dest='recall', type=str, required=False, default='',
                        help='A pickled python object for computing the power of detecting IBD segments at various segment length.\
                        We use scipy.interpolate.interp1d, but any pickled callable object that takes exactly one argument (segment length) and returns the recall at that length will work.')
    parser.add_argument('--lenbias', action='store', dest='lenbias', type=str, required=False, default='',
                        help='A pickled python object for computing the length bias of IBD segments. \
                        The function f(x) should give the probability density function of a true IBD segment of length l being inferred as l+x (Note that x can be negative).\
                        We use scipy.stats.gaussian_kde, but any pickled callable object that takes exactly one argument (length bias) and returns the pdf evaluated at the function argument will work.')
    parser.add_argument('--dryrun', action='store_true', dest='dryrun', required=False, default=False, 
                        help='If set, the script will only cluster sample into multiple sample points based on the date information and report such information, \
                            without actually inferring the Ne. This is useful for sanity check and for selecting the appropriate --minN value.')
    parser.add_argument('--boot', action='store_true', dest='boot', required=False, default=False, 
                        help='If set, the script will perform bootstrapping by resampling chromosomes. Default is False.')
    parser.add_argument('--autocv', action='store_true', dest='autocv', required=False, default=False, 
                        help='If set, the script will perform automatic cross-validation to select the optimal level of regularization. Default is False.')
    parser.add_argument('--np', action='store', dest='np', type=int, required=False, default=1, 
                        help='The number of parallel processes to use. Default is 1. \
                        If np>1, the script will use the multiprocessing module to parallelize the computation. We strongly recommend you to set this value to >1 \
                        if any of the --boot or --autocv is set to true.')
    parser.add_argument('--prefix', action='store', dest='prefix', type=str, required=False, default='',
                        help='The prefix of the output file. Default is empty string.')
    parser.add_argument('--out', action='store', dest='out', type=str, required=False, default='',
                        help='The output directory. Default is the current working directory.')
    args = parser.parse_args()
    

    if len(args.out) == 0:
        # set outfolder to cwd
        args.out = os.getcwd()

    if len(args.fp) == 0 or len(args.recall) == 0 or len(args.lenbias) == 0:
        FP = R = POWER = None
    else:
        minL_calc = args.minl_calc
        maxL_calc = args.maxl_calc
        step = 0.25
        bins = np.arange(minL_calc, maxL_calc+step, step)
        binmidpoint = (bins[1:]+bins[:-1])/2
        with open(args.fp, 'rb') as f:
            FP_func = pickle.load(f)
            FP = FP_func(binmidpoint)
        with open(args.recall, 'rb') as f:
            kde = pickle.load(f)
            lengthDiff = binmidpoint.reshape(1, -1) - binmidpoint.reshape(-1, 1)
            nrow, ncol = lengthDiff.shape
            R = np.zeros_like(lengthDiff)
        for i in range(nrow):
            for j in range(ncol):
                R[i,j] = step*kde.evaluate(lengthDiff[i,j])
        with open(args.lenbias, 'rb') as f:
            POWER_func = pickle.load(f)
            POWER = POWER_func(binmidpoint)

    inferVecNe_singlePop_MultiTP_withMask(args.path2IBD, args.path2SampleAge, args.chrdelim, path2mask=args.path2mask,
        Tmax=args.Tmax, alpha=args.alpha, beta=250, method='l2', 
        minL_calc=args.minl_calc, maxL_calc=args.maxl_calc, minL_infer=args.minl_infer, maxL_infer=args.maxl_infer, step=0.25,
        FP=FP, R=R, POWER=POWER, generation_time=args.generation_time, minSample=args.minN, merge_level=args.merge, 
        prefix="", doBootstrap=args.boot, autoHyperParam=args.autocv, outFolder=args.out, dryrun=args.dryrun, parallel=args.np>1, nprocess=args.np)