# [Scalable Orthonormal Projective NMF via Diversified Stochastic Optimization](https://doi.org/10.1007/978-3-031-34048-2_38)

## Introduction

This is the accompanying python code for the 2023 Information Processing in Medical Imaging (IPMI) Conference manuscript "[ Scalable Orthonormal Projective NMF via Diversified Stochastic Optimization](https://doi.org/10.1007/978-3-031-34048-2_38)". This implementation uses stochastic optimization using either random uniform sampling of determinantal point processes (DPP) to perform stochastic learning with orthonormal projective Nonnegative Matrix Factorization (opNMF from [Linear and Nonlinear Projective Nonnegative Matrix Factorization](https://doi.org/10.1109/TNN.2010.2041361)), reducing the memory footprint of the method and improving its scalability to big data. The opNMF implementation for neuroimaging context is a stripped down port of the matlab `opnmf.m` and `opnmf_mem.m` codes found at [brainparts github repository](https://github.com/asotiras/brainparts) to python. Portions of the DPP implementations were adapted matlab implementation at [Dr. Alex Kulesza's website](https://www.alexkulesza.com/), python implementation of DPP in [HYDRA github repository](https://github.com/evarol/HYDRA), and python implementation of DPP inside Hydra found at [MLNI github repository](https://github.com/anbai106/mlni).

## Prerequisites

This code was tested on Linux (__Rocky Linux release 8.8__) operating system using python installed through [anaconda](https://www.anaconda.com/). The following python packages, available either through the default repository or the conda-forge repository, are required to run this code.

| Package Name | Package Version Tested | Notes |
| :----------: | :--------------------: | :---- |
| numpy | 1.20.3 |  |
| scipy | 1.7.1 | |
| scikit-learn | 1.0.1 | |
| hdf5storage | 0.1.16 | this is used to save and load input, intermediate, and final output files in compressed hdf5 format that can also be loaded in MatLab. |

## Quick Start

### opNMF (non-stochastic)

1. You will need to prepare a __nonnegative__ input data matrix in hdf5 compressed format with `.mat` extension, saved with variable name `X`.
1. For example, you can create a random data matrix of size of size 5000 by 1000, save it to a `input.mat` file, using the following python snippet.
    ```
    import numpy as np
    import hdf5storage

    X = np.abs(np.random.rand(5000, 1000))
    hdf5storage.savemat("./input.mat", mdict = {"X": X})
    ```
    1. Note that unlike [brainparts github repository](https://github.com/asotiras/brainparts) implementation that has additional preprocessing to remove common zero pixels/voxels across all columns and to downsample the `X` matrix prior to multiplicative updates, this implementation does NOT have such preprocessing. The appropriate preprocessing and downsampling is left to the end user to carry out prior to calling this code snippet.
    1. Also note that the input data matrix `X` has to be __non-negative__!
1. Determine how many components you want to generate? We will call this value target rank. Default is $20$.
1. Determine how many iterations to run before terminating the multiplicative updates. Default is $1.0 \times 10^4$.
1. Determine what early stopping criterion threshold to use such that if $` \frac{{(\| {W}_{t+1} - {W}_{t} \|)}^{2}_{F} }{ {(\| W_{t+1} \|)}^{2}_{F} } < tol `$.
    * If the condition is met, the update will terminate. Default is $1.0 \times 10^{-6}$.
1. Determine how you want your initial component matrix $W$ to be intialized. Default is to use [nndsvd](https://doi.org/10.1016/j.patcog.2007.09.010).
1. Determine where the outputs will be generated.
1. Call the `Python/opnmf.py` script with appropriate parameters.
    1. for opNMF with speed optimized but high memory footprint resource usage:
    ```
    python opnmf.py --inputFile="/path/to/input.mat" --outputParentDir="/path/to/output/directory" --targetRank=20 --initMeth="nndsvd" --maxIter=10000 --updateMeth="original" --tol=0.000001
    ```
    1. for opNMF with memory optimized but speed inefficient resource usage:
    ```
    python opnmf.py --inputFile="/path/to/input.mat" --outputParentDir="/path/to/output/directory" --targetRank=20 --initMeth="nndsvd" --maxIter=10000 --updateMeth="mem" --tol=0.000001
    ```

### sopNMF (stochastic) / dsopNMF (DPP stochastic)
1. You will need to prepare a __nonnegative__ input data matrix in hdf5 compressed format with `.mat` extension, saved with variable name `X`.
1. For example, you can create a random data matrix of size of size 5000 by 1000, save it to a `input.mat` file, using the following python snippet.
    ```
    import numpy as np
    import hdf5storage

    X = np.abs(np.random.rand(5000, 1000))
    hdf5storage.savemat("./input.mat", mdict = {"X": X})
    ```
    1. Note that unlike [brainparts github repository](https://github.com/asotiras/brainparts) implementation that has additional preprocessing to remove common zero pixels/voxels across all columns and to downsample the `X` matrix prior to multiplicative updates, this implementation does NOT have such preprocessing. The appropriate preprocessing and downsampling is left to the end user to carry out prior to calling this code snippet.
    1. Also note that the input data matrix `X` has to be __non-negative__!
1. Determine how many components you want to generate? We will call this value target rank. Default is $20$.
1. Determine how many iterations to run before terminating the multiplicative updates. Default is $1.0 \times 10^4$.
1. Determine what early stopping criterion threshold to use such that if $` \frac{{(\| {W}_{t+1} - {W}_{t} \|)}^{2}_{F} }{ {(\| W_{t+1} \|)}^{2}_{F} } < tol `$.
    * If the condition is met, the update will terminate. Default is $1.0 \times 10^{-6}$.
1. Determine how you want your initial component matrix $W$ to be intialized. Default is to use [nndsvd](https://doi.org/10.1016/j.patcog.2007.09.010).
1. Determine where the outputs will be generated.
1. Determine how big you want your mini-batch to be. Default value is $p=$ `batch_size` $= 100$.
    * Note that current implementation does not take into account the edge case scenario where the total number of subjects $n$ (= number of subjects = number of columns of `X`) divided by the number of subjects in mini-batch $p$ results in dividend other than 0.
1. Determine what sampling method you want to use to sample the mini-batches.
    1. `--samplingMeth=uniform`: uniform random sampling
        1. If you want uniform random sampling __with__ replacement, make sure to also call the flag `--withReplacement`
        1. If you do not call `--withReplacement` flag, then it will perform uniform random sampling __without__ replacement.
    1. `--samplingMeth=dpplinear`: Using DPP on $ X^{T}X $ as the kernel.
    1. `--samplingMeth=dppgaussian`: Using DPP on some csv file of $ n + 1 $ for $ X \in \mathbb{R}^{(m \times n)} $ with $n$ subjects. The file expects two variables as headers of the csv file, `age` and `sex`. The csv file path should be provided by `--demographic_data_path=/path/to/file.csv` argument.
        1. Use `--sigma` argument to provide $ \sigma $ for the Gaussian kernel.
1. Call the `Python/sopnmf.py` script with appropriate parameters.
    1. for sopNMF with uniform sampling without replacement:
    ```
    python sopnmf.py --inputFile="/path/to/input.mat" --outputParentDir="/path/to/output/directory" --targetRank=20 --initMeth="nndsvd" --maxEpoch=10000 --multiplicativeUpdateMeth="normalize" --tol=0.000001 --sampleSize=100 --samplingMeth="uniform" --updateMeth="mem"
    ```
    1. for sopNMF with uniform sampling with replacement:
    ```
    python sopnmf.py --inputFile="/path/to/input.mat" --outputParentDir="/path/to/output/directory" --targetRank=20 --initMeth="nndsvd" --maxEpoch=10000 --multiplicativeUpdateMeth="normalize" --tol=0.000001 --sampleSize=100 --samplingMeth="uniform" --withReplacement --updateMeth="mem"
    ```
    1. for dsopNMF with DPP gaussian sampling on demographic data:
    ```
    python sopnmf.py --inputFile="/path/to/input.mat" --outputParentDir="/path/to/output/directory" --targetRank=20 --initMeth="nndsvd" --maxEpoch=10000 --multiplicativeUpdateMeth="normalize" --tol=0.000001 --sampleSize=100 --samplingMeth="dppgaussian" --sigma=0.1 --demographic_data_path="/path/to/demographic/age/sex/data.csv" --updateMeth="mem"
    ```
    1. for dsopNMF with DPP linear sampling on $X^{T}X$:
    ```
    python sopnmf.py --inputFile="/path/to/input.mat" --outputParentDir="/path/to/output/directory" --targetRank=20 --initMeth="nndsvd" --maxEpoch=10000 --multiplicativeUpdateMeth="normalize" --tol=0.000001 --sampleSize=100 --samplingMeth="dpplinear" --updateMeth="mem"
    ```
