# TTNe (Time Transect Ne Estimation)

TTNe is a method for estimating recent effective population size (_Ne_) using samples belonging to the same population but dated to different time periods. It is mainly desgiend for ancient DNA data where time transect sampling is becoming more common in some of the densely studied regions. It takes as input the detected IBD segments (obtained from, for example, [ancIBD](https://www.nature.com/articles/s41588-023-01582-w)) among the samples and then infers a Ne trajectory consistent with the observed IBD patterns using a maximum likelihood approach.

## Installation

todo

## Basic Usage

To use TTNe, you need to first call IBD segments for your data. You can first use GLIMPSE (or other low-coverage imputation tools if you prefer) to impute your aDNA data and then use ancIBD (or other IBD callers if you prefer) to obtain IBD segments. We have prepared a small test dataset to demonstrate the basic usage of TTNe. The test dataset is in the folder ./test/input.

The test dataset was a bottleneck demography simulated by [msprime](https://tskit.dev/msprime/docs/stable/intro.html). It consists of a total of 120 samples taken at 2 time points (t=0 and t=20 generations backward in time). The bottleneck event happens at 30^th generations backward in time. 

The file ./test/input/BN.TP2.8cM.ibd.tsv contains IBD segments longer than 8cM extracted from the simulated ARG. It has four columns: iid1, iid2, ch and lengthM. Note that the segment length is in Morgen. The input .tsv file for IBD segments can also contain additional columns but it should at least contain these four columns (but see below if apply a genomic mask, in which case additional columns are needed).

The file ./test/input/BN.TP2.sample.age contains date information. You should provide dating information for all samples in the IBD .tsv file, one sample per line. The date should be expressed in "years BP". In this test dataset, the first 60 samples are sampled from the 0<sup>th</sup> generation thus having the date 0. The last 60 samples are sampled from the 20<sup>th</sup> generations and therefore are dated to 580 years ago, assuming a generation time of 29 years.

The file ./test/input/chr.all.start_end specifies the genetic map position of the first and last SNP used for IBD calling. It is used for calculating the total genetic map length and also for working with genomic masks (see below).

These three files are the minimum input files for TTNe. With that, we can run the following command to test it out, assuming your current working directory is the ./test folder. The following command uses IBD segments within the length range 8-20cM (--minl_infer and --maxl_infer) to infer Ne within the 100 generations backward (--Tmax) from the oldest samples.

```
    TTNe --IBD ./input/BN.TP2.8cM.ibd.tsv --chrdelim ./input/chr.all.start_end --date ./input/BN.TP2.sample.age --Tmax 100 --minl_infer 8 --maxl_infer 20 --out output
```

You will get three files in the output folder (--out). The Ne.csv file gives the inferred Ne at each generation. The pairwise.fit.png shows the fitting between the observed IBD segments and the expected IBD sharing calculated from the inferred Ne trajectory. The pairwise.postTMRCA_cdf.png shows the CDF of TMRCA of IBD segments under the inferred Ne trajectory. For the sake of runtime, the above command does not run the cross validation and bootstrap (see our manuscript for details). To enable these two (recommend!), we add two flags to the command,

```
    TTNe --IBD ./input/BN.TP2.8cM.ibd.tsv --chrdelim ./input/chr.all.start_end --date ./input/BN.TP2.sample.age --Tmax 100 --minl_infer 8 --maxl_infer 20 --out output --autocv --bootstrap
```
This takes about 40min when using a single process. You can set --np to use multiple processes to speed up. You can view all the available command line options by typing `TTNe -h`.

## Specifying Error Model

todo





 