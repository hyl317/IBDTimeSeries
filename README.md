# TTNe (Time Transect Ne Estimation)

TTNe is a method for estimating recent effective population size (_Ne_) using samples belonging to the same population but dated to different time periods. It is mainly desgiend for ancient DNA data where time transect sampling is becoming more common in some of the densely studied regions. It takes as input the detected IBD segments (obtained from, for example, [ancIBD](https://www.nature.com/articles/s41588-023-01582-w)) and then infers a Ne trajectory consistent with the observed IBD patterns using a maximum likelihood approach.

## Installation

pip install TTNe

## Basic Usage

To use TTNe, you need to first call IBD segments for your data. You can first use GLIMPSE (or other low-coverage imputation tools if you prefer) to impute your aDNA data and then use ancIBD (or other IBD callers if you prefer) to obtain IBD segments. We have prepared a small test dataset to demonstrate the basic usage of TTNe. The test dataset is in the folder ./test/input.

The test dataset is a bottleneck demography simulated by [msprime](https://tskit.dev/msprime/docs/stable/intro.html). It consists of 120 samples taken at 2 time points (t=0 and t=20 generations backward in time). The bottleneck event happens at t=30. 

The file ./test/input/BN.TP2.8cM.ibd.tsv contains IBD segments longer than 8cM extracted from the simulated ARG. It has four columns: iid1, iid2, ch and lengthM. These four mandatory columns are present in the ancIBD output. For IBD called by other software, you may need to reformat the file. Note that the segment length is in Morgen. The input .tsv file for IBD segments can also contain additional columns but it should at least contain these four columns (but see below if apply a genomic mask, in which case additional columns are needed). 

The file ./test/input/BN.TP2.sample.age contains date information. You should provide dating information for all samples in the IBD .tsv file, one sample per line. The date should be expressed in "years BP". In this test dataset, the first 60 samples are sampled from the 0<sup>th</sup> generation thus having the date 0. The last 60 samples are sampled from the 20<sup>th</sup> generations and therefore are dated to 580 years ago, assuming a generation time of 29 years.

The file ./test/input/chr.all.start_end specifies the genetic map position of the first and last SNP used for IBD calling. It is used for calculating the total genetic map length and also for working with genomic masks (see below).

These three files are the minimum input files for TTNe. With that, we can run the following command to test it out, assuming your current working directory is the ./test folder. The following command uses IBD segments within the length range 8-20cM (--minl_infer and --maxl_infer) to infer Ne within the 100 generations backward (--Tmax) from the oldest samples.

```
    TTNe --IBD ./input/BN.TP2.8cM.ibd.tsv --chrdelim ./input/chr.all.start_end --date ./input/BN.TP2.sample.age --Tmax 100 --minl_infer 8 --maxl_infer 20 --out output
```

You will get three files in the output folder (--out). The Ne.csv file gives the inferred Ne at each generation. The pairwise.fit.png shows the fitting between the observed IBD segments (grey dots) and the expected IBD sharing (red line) calculated from the inferred Ne trajectory. In a successful fitting, you should see the grey dots fall closely around the red line. The pairwise.postTMRCA_cdf.png shows the CDF of TMRCA of IBD segments under the inferred Ne trajectory. For the sake of runtime, the above command does not run the cross validation and bootstrap (see our manuscript for details). To enable these two (recommend!), we add two flags (--autocv and --boot) to the command,

```
    TTNe --IBD ./input/BN.TP2.8cM.ibd.tsv --chrdelim ./input/chr.all.start_end --date ./input/BN.TP2.sample.age --Tmax 100 --minl_infer 8 --maxl_infer 20 --out output --autocv --boot
```
This takes about 40min when using a single process. You can set --np to use multiple processes to speed up. You can view all the available command line options by typing `TTNe -h`. There are several parameters worthing considering tuning/altering when you apply TTNe to your data.

- minl_infer: IBD length cutoffs. We recommend at least 8cM. You can experiment with more strigent cutoffs to see whether the inferred Ne trajectories are consistent. It is expected that with fewer IBD segments, the power of detecting population size changes is significantly reduced.

- merge: This parameter controls the degree to which samples of similar dates will be merged into a single time point. This is a necessary step because, although our model still works when all samples have distinct dates, this creates a huge computational burden because the likelihood evaluation takes O(n<sup>2</sup>) time, where n is the number of distinct time points. By default, we merge the sample by 5 generations. You can also adjust the generation time by --generation_time as you see fit. The default is 29 years. You may also wish to adjust --minN, the minimum sample size required for a time point to be included. You can eliminate time points with very few samples as they probably do not add much information but significantly increase runtime. The default is 10.

## Use of genomic masks

You may wish to mask certain genomic regions for various reasons (e.g, low SNP density leading to degrading IBD detection performance, positively selected regions etc). In that case, you can supply a genomic mask file. An example mask file can be found at ./test/input/mask.track. It's a tab-separated file and each masked region occupies one line. The first column is the chromosome ID, the second/third columns are the basepair positions of the start and end of the mask, and the fourth/fifth columns are the genetic map position (in cM) of the mask. If mask is provided, the IBD .tsv file must also contain two additional columns "StartM" and "EndM", the start and end genetic map position (in Morgen) of the IBD segments. You supply the mask file via the option --mask.

## Specifying Error Model

This is an advanced topic and is not trivial to adapt to everyone's specific case unfortunately. At a high level, one needs to simulate IBD and evaluate IBD caller's precision, recall and length bias at the coverage typical for your samples. In our manuscript, we simulated IBD by using the naturally occuring IBD between parent-offspring pairs. To simulate an IBD block shared between two samples, we take reads aligned to the intended IBD block from the parent and offspring BAM files, respectively. For non-IBD block, we take aligned reads from another two unrelated samples. For more details, please refer to Supplementary Note 2 of our manuscript.

You can specify the error model via --fp, --recall, --lenbias. All of them expect a pickled python object that takes in a single parameter x or a numpy array and returns corresponding values for the error model (see descriptions below).

- False positives:
    The false positive rate function should take as argument IBD length (in cM), either as a float point number or as a numpy array, and return the density of false positive rate x cM long per pair of haplotypes (not a pair of diploid samples!) and per cM. This definition might be a bit abstract. More concretely, suppose we have n false positive IBD segments falling into a small length bin of width 0.25cM (say, between 7.875-8.125cM), among $N$ pairs of haplotypes of length lcM. Then the density of false positive rate of 8cM IBD segments can be approximated by $\frac{n}{(N*l*0.25)}$. 

- Power:
    The power function should take as argument IBD length (in cM), either as a plain float point number or as a numpy array, and return the power of detecting IBD segments of that length. 

- Length Bias:
    Length bias is the inferred IBD length minus the groundtruth IBD length. We make the simplifying assumption that length bias is independent of the groundtruth IBD length. We expect --lenbias to be a pickled scipy.stats.gaussian_kde() object. We use scipy.stats.gaussian_kde.evaluate() function to evaluate the probability density function of length bias. More precisely, let y=gaussian_kde.evaluate(x), then $y \delta l$ is the probability that a IBD segment of length $l$ will have inferred length within the small interval $[l+x-\frac{\delta l}{2},l+x+\frac{\delta l}{2}]$.

> [!NOTE]
> If an error model is specified, it is important to set --minl_calc and --maxl_calc. This is because, with IBD length bias, shorter segments may be observed as longer segments (and analogously, longer segments may be observed as shorter). Therefore, it is necessary to consider a wider length range than the IBD length range actually used for inference (which is specified by --minl_infer and --maxl_infer).




 