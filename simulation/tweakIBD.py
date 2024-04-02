import argparse
import sys
import pickle
import os
import numpy as np
from ibdDemo.simulation.tweakIBD_helper import FP, POWER, alterIBD_multiTP, power, power2

ch_len_dict = {1:286.279, 2:268.840, 3:223.361, 4:214.688, 5:204.089, 6:192.040, 7:187.221, 8:168.003, 9:166.359, \
        10:181.144, 11:158.219, 12:174.679, 13:125.706, 14:120.203, 15:141.860, 16:134.038, 17:128.491, 18:117.709, \
        19:107.734, 20:108.267, 21:62.786, 22:74.110}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tweak groundtruth IBD according to an error model')
    parser.add_argument('--ibd', action="store", dest="ibd", type=str, required=True,
                        help="path to the groundtruth IBD pickle file")
    parser.add_argument('-n', action="store", dest="n", type=str, required=True,
                        help="number of diploid samples to take at each time point")
    parser.add_argument('-t', action="store", dest="t", type=str, required=True,
                        help="sampling time of each sampling clusters. Must be in the same order as given in -n.")
    parser.add_argument('-r', action="store", dest="r", type=int, required=False, default=10,
                        help="replicate index.")
    parser.add_argument('--out', action="store", dest="out", type=str, required=True,
                        help="output directory.")
    args = parser.parse_args()

    r = args.r
    outFolder = args.out
    nSamples = [int(n) for n in args.n.split(',')]
    gaps = [int(g) for g in args.t.split(',')]
    gaps_ = {}
    for i, gap in enumerate(gaps):
        gaps_[i] = gap
    nSamples_ ={}
    for i, nSample in enumerate(nSamples):
        nSamples_[i] = nSample

    # infer Ne from tweaked IBD
    sys.path.append("/mnt/archgen/users/yilei/IBD/timeSampling")
    from ibdDemo.analytic_multi import inferVecNe_singlePop_MultiTP
    

    ############################################### test length bias error ########################################################
    # loc = 0.0
    # sigma = 1.5
    # # alterIBD(args.ibd, nSamples, ch_len_dict, POWER=None, FP=None, loc=loc, scale=sigma, suffix='error_sigma3over2')
    # ibds = pickle.load(open(f'{args.ibd}.error_sigma3over2', 'rb'))
    
    # # Ne, lowCIs, upCIs = inferVecNe_singlePop_MultiTP(ibds, gaps_, nSamples_, ch_len_dict, Ninit=500, Tmax=50, \
    # #     minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, step=0.1, alpha=500, beta=0, method='l2')
    # # with open(f'{outFolder}/rep{r}.vecNe.l2.alpha_500.beta_0.6cM.error_sigma3over2_noCorr', 'w') as out:
    # #     for i, tuple in enumerate(zip(Ne, lowCIs, upCIs)):
    # #         n, lowCI, upCI = tuple
    # #         out.write(f'{i+1}\t{round(n,3)}\t{round(lowCI,3)}\t{round(upCI,3)}\n')

    # bins_calc = np.arange(1.5, 24.5+0.1, 0.1)
    # binMidpoint_calc = (bins_calc[1:] + bins_calc[:-1])/2
    # nBins = len(binMidpoint_calc)
    # lengthDiff = binMidpoint_calc.reshape(nBins, 1) - binMidpoint_calc.reshape(1, nBins)
    # FPVec = np.zeros(nBins)
    # POWERVec = np.ones(nBins)
    # R = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*np.square((lengthDiff-loc)/sigma))
    # R = 0.1*R
    # Ne, lowCIs, upCIs = inferVecNe_singlePop_MultiTP(ibds, gaps_, nSamples_, ch_len_dict, Ninit=500, Tmax=50, \
    #     minL_calc=1.5, maxL_calc=24.5, minL_infer=6.0, maxL_infer=20.0, step=0.1, alpha=500, beta=0, method='l2', \
    #     FP=FPVec, R=R, POWER=POWERVec)
    # with open(f'{outFolder}/rep{r}.vecNe.l2.alpha_500.beta_0.6cM.error_sigma3over2_corr', 'w') as out:
    #     for i, tuple in enumerate(zip(Ne, lowCIs, upCIs)):
    #         n, lowCI, upCI = tuple
    #         out.write(f'{i+1}\t{round(n,3)}\t{round(lowCI,3)}\t{round(upCI,3)}\n')
    ############################################### end of test length bias error ########################################################


    ############################################### test FP error ################################################
    # loc = 0.0
    # sigma = 0
    # FPscale = 100
    # FPVec = FPscale*FP(binMidpoint)
    # R = np.eye(nBins) # null model for length bias, to isolate the effect of FP
    # POWERVec = np.ones(nBins) # null model for power, to isolate the effect of FP
    # if not os.path.isfile(f'{args.ibd}.error_FP100'):
    #     alterIBD(args.ibd, nSamples, ch_len_dict, FP=FP, FPscale=FPscale, loc=loc, scale=sigma, suffix='error_FP100')
    # ibds = pickle.load(open(f'{args.ibd}.error_FP100', 'rb'))
    
    # print('inference with no FP correction')
    # Ne, lowCIs, upCIs = inferVecNe_singlePop_MultiTP(ibds, gaps_, nSamples_, ch_len_dict, Ninit=500, Tmax=50, minL=6.0, maxL=20.0, step=0.1, alpha=500, beta=0, method='l2')
    # with open(f'{outFolder}/rep{r}.vecNe.l2.alpha_500.beta_0.6cM.error_FP100_noCorr', 'w') as out:
    #     for i, tuple in enumerate(zip(Ne, lowCIs, upCIs)):
    #         n, lowCI, upCI = tuple
    #         out.write(f'{i+1}\t{round(n,3)}\t{round(lowCI,3)}\t{round(upCI,3)}\n')

    # print('inference with FP correction')
    # Ne, lowCIs, upCIs = inferVecNe_singlePop_MultiTP(ibds, gaps_, nSamples_, ch_len_dict, Ninit=500, Tmax=50, minL=6.0, maxL=20.0, step=0.1, alpha=500, beta=0, method='l2', FP=FPVec, R=R, POWER=POWERVec)
    # with open(f'{outFolder}/rep{r}.vecNe.l2.alpha_500.beta_0.6cM.error_FP100_corr', 'w') as out:
    #     for i, tuple in enumerate(zip(Ne, lowCIs, upCIs)):
    #         n, lowCI, upCI = tuple
    #         out.write(f'{i+1}\t{round(n,3)}\t{round(lowCI,3)}\t{round(upCI,3)}\n')

    ############################################### end of test FP error ################################################

    ############################################## Test of imperfect power #############################################
    # loc = 0.0
    # sigma = 0
    # alterIBD(args.ibd, nSamples, ch_len_dict, POWER=power2, FP=None, loc=loc, scale=sigma, suffix='error_power2')
    # ibds = pickle.load(open(f'{args.ibd}.error_power2', 'rb'))
    
    # Ne, lowCIs, upCIs = inferVecNe_singlePop_MultiTP(ibds, gaps_, nSamples_, ch_len_dict, Ninit=500, Tmax=50, \
    #     minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, step=0.1, alpha=500, beta=0, method='l2')
    # with open(f'{outFolder}/rep{r}.vecNe.l2.alpha_500.beta_0.6cM.error_power2_noCorr', 'w') as out:
    #     for i, tuple in enumerate(zip(Ne, lowCIs, upCIs)):
    #         n, lowCI, upCI = tuple
    #         out.write(f'{i+1}\t{round(n,3)}\t{round(lowCI,3)}\t{round(upCI,3)}\n')

    # bins_calc = np.arange(6, 20+0.1, 0.1)
    # binMidpoint_calc = (bins_calc[1:] + bins_calc[:-1])/2
    # nBins = len(binMidpoint_calc)
    # FPVec = np.zeros(nBins)
    # POWERVec = power2(binMidpoint_calc)
    # R = np.eye(nBins)
    # Ne, lowCIs, upCIs = inferVecNe_singlePop_MultiTP(ibds, gaps_, nSamples_, ch_len_dict, Ninit=500, Tmax=50, \
    #     minL_calc=6, maxL_calc=20, minL_infer=6.0, maxL_infer=20.0, step=0.1, alpha=500, beta=0, method='l2', \
    #     FP=FPVec, R=R, POWER=POWERVec)
    # with open(f'{outFolder}/rep{r}.vecNe.l2.alpha_500.beta_0.6cM.error_power2_corr', 'w') as out:
    #     for i, tuple in enumerate(zip(Ne, lowCIs, upCIs)):
    #         n, lowCI, upCI = tuple
    #         out.write(f'{i+1}\t{round(n,3)}\t{round(lowCI,3)}\t{round(upCI,3)}\n')
    ########################################## end of test of imperfect power ########################################

    ####################################  test everything combined ############################
    loc = 0.0
    sigma = 1.5
    # alterIBD_multiTP(args.ibd, nSamples, ch_len_dict, POWER=power2, FP=FP, FPscale=100, loc=loc, scale=sigma, suffix='error_combined')
    ibds = pickle.load(open(f'{args.ibd}.error_combined', 'rb'))
    
    # Ne, lowCIs, upCIs = inferVecNe_singlePop_MultiTP(ibds, gaps_, nSamples_, ch_len_dict, Ninit=500, Tmax=50, \
    #     minL_calc=2.0, maxL_calc=24.0, minL_infer=6.0, maxL_infer=20.0, step=0.1, alpha=500, beta=0, method='l2')
    # with open(f'{outFolder}/rep{r}.vecNe.l2.alpha_500.beta_0.6cM.error_combined_noCorr', 'w') as out:
    #     for i, tuple in enumerate(zip(Ne, lowCIs, upCIs)):
    #         n, lowCI, upCI = tuple
    #         out.write(f'{i+1}\t{round(n,3)}\t{round(lowCI,3)}\t{round(upCI,3)}\n')

    bins_calc = np.arange(1.5, 24.5+0.1, 0.1)
    binMidpoint_calc = (bins_calc[1:] + bins_calc[:-1])/2
    nBins = len(binMidpoint_calc)
    lengthDiff = binMidpoint_calc.reshape(nBins, 1) - binMidpoint_calc.reshape(1, nBins)
    FPVec = 100*FP(binMidpoint_calc)
    POWERVec = power2(binMidpoint_calc)
    R = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*np.square((lengthDiff-loc)/sigma))
    R = 0.1*R
    Ne, lowCIs, upCIs = inferVecNe_singlePop_MultiTP(ibds, gaps_, nSamples_, ch_len_dict, Ninit=500, Tmax=75, \
        minL_calc=1.5, maxL_calc=24.5, minL_infer=6.0, maxL_infer=20.0, step=0.1, alpha=2500, beta=0, method='l2', \
        FP=FPVec, R=R, POWER=POWERVec, plot=True, outFolder=outFolder, prefix=f'rep{r}.error_combined')
    with open(f'{outFolder}/rep{r}.vecNe.l2.alpha_2500.beta_0.6cM.error_combined_corr', 'w') as out:
        for i, tuple in enumerate(zip(Ne, lowCIs, upCIs)):
            n, lowCI, upCI = tuple
            out.write(f'{i+1}\t{round(n,3)}\t{round(lowCI,3)}\t{round(upCI,3)}\n')


    ################################### end of test of everything combined ######################