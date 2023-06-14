"""
s-opNMF (stochastic opNMF) implemented in python

* 2 Methods of sampling is available for stochastic optimization
    * Uniform Sampling
        * Uniform Sampling
        * without replacement
    * DPP Sampling
        * with Gaussian Kernel
        * with Linear Kernel
"""

## Module Imports
import sys #check python version, exit upon sanity check failures
import os #path checks
import numpy as np #matrix multiplications and optimizations
import argparse #for taking in user input/parameter/switch
import time #for measuring elapsed time
import datetime #for default output path string formation
import hdf5storage #for saving as hdf5 mat file (v7.3 on matlab); has much better compression than scipy savemat
import getpass  #for getting username on compute jobs since os.getlogin() works in interactive jobs but not compute jobs
import pandas as pd #for reading csv
import shutil #for renaming files

import utils #for loading and saving data
import initialize_nmf #for W component initialization
import opnmf_update_rule #for W component initialization

#default value for script name to be used for printing to console
script_name=os.path.basename(__file__)
username=str(getpass.getuser())

## FLAGS
VERBOSE_FLAG = False
DEBUG_FLAG = False
WITH_REPLACEMENT_FLAG = False

## Version
script_version = "20221115_135100"

#sanity check: python version
utils.check_python_version(major = 3, minor = 7) #ensure the user is using python3.7 or higher

## Default values
csv_path = os.path.join("/scratch/sungminha/git/NMF_Testing_Preprocessing/output_directory/dramms_1000_subjectid_only_those_without_negative_values_or_bad_ICV_changes_fullpaths_subjectid_age_sex_cdr_with_header.csv")
feret_path = os.path.join("/scratch/sungminha/git/NMF_Testing/faces_1024_2409_min0_max1_double.mat")
EPSILON = np.finfo(np.float32).eps


## s-opNMF specific functions
def sample_uniform(X,  num_batch, batch_size, max_epoch, axis = 1, WITH_REPLACEMENT = False):
    """
    inputs
    X: input matrix of size [m,n]
    axis (default=1): which direction to sample from; default is n in X in [m,n] where n is assumed to be subjects
    """
    #assign indices from 0 to n (where n is number of subjects, or number of columns of X)
    idx_list = np.arange(start = 0, stop = np.shape(X)[axis], step = 1)

    idx_sample_full = -1 * np.ones(shape = (batch_size, num_batch, max_epoch)) #since indices are nonnegative, initialize indices to bogus value of negative one so that it is easy to check for mistake when debugging by checking if any value of idx_sample_full is negative
    for epoch in np.arange(max_epoch):
        if VERBOSE_FLAG:
            print("%s: sample_uniform: sampling %i-th epoch indices." % (script_name, epoch), flush=True)
        idx_sample_full[:,:,epoch] = np.random.choice(idx_list, size = (batch_size, num_batch), replace = WITH_REPLACEMENT)
    return idx_sample_full

def sample_dpp(evalue, evector, k):
    """
    sample a set Y from a dpp.  evalue, evector are a decomposed kernel, and k is (optionally) the size of the set to return
    :param evalue: eigenvalue
    :param evector: normalized eigenvector
    :param k: number of cluster
    :return:
    """
  
    if k == None:
        # choose eigenvectors randomly
        evalue = np.divide(evalue, (1 + evalue))
        v = np.where(np.random.random(evalue.shape[0]) <= evalue)[0]
        #evector = np.where(np.random.random(evalue.shape[0]) <= evalue)[0]
    else:
        v = sample_k(evalue, k) ## v here is a 1d array with size: k

    k = v.shape[0]
    v = v.astype(int)
    v = [i - 1 for i in v.tolist()]  ## due to the index difference between matlab & python, here, the element of v is for matlab
    V = evector[:, v]

    ## iterate
    y = np.zeros(k)
    for i in range(k, 0, -1):
        ## compute probabilities for each item
        P = np.sum(np.square(V), axis=1)
        P = P / np.sum(P)

        # choose a new item to include
        y[i-1] = np.where(np.random.rand(1) < np.cumsum(P))[0][0]
        y = y.astype(int)

        # choose a vector to eliminate
        j = np.where(V[y[i-1], :])[0][0]
        Vj = V[:, j]
        V = np.delete(V, j, 1)

        ## Update V
        if V.size == 0:
           pass
        else:
            V = np.subtract(V, np.multiply(Vj, (V[y[i-1], :] / Vj[y[i-1]])[:, np.newaxis]).transpose())  ## watch out the dimension here

        ## orthogonalize
        for m in range(i - 1):
            for n in range(m):
                V[:, m] = np.subtract(V[:, m], np.matmul(V[:, m].transpose(), V[:, n]) * V[:, n])

            V[:, m] = V[:, m] / np.linalg.norm(V[:, m])

    y = np.sort(y)

    return y

def sample_k(lambda_value, k):
    """
    Pick k lambdas according to p(S) \propto prod(lambda \in S)
    :param lambda_value: the corresponding eigenvalues
    :param k: the number of clusters
    :return:
    """

    ## compute elementary symmetric polynomials
    E = elem_sym_poly(lambda_value, k)

    ## ietrate over the lambda value
    num = lambda_value.shape[0]
    remaining = k
    S = np.zeros(k)
    while remaining > 0:
        #compute marginal of num given that we choose remaining values from 0:num-1
        if num == remaining:
            marg = 1
        else:
            marg = lambda_value[num-1] * E[remaining-1, num-1] / E[remaining, num]

        # sample marginal
        if np.random.rand(1) < marg:
            S[remaining-1] = num
            remaining = remaining - 1
        num = num - 1
    return S

def elem_sym_poly(lambda_value, k):
    """
    given a vector of lambdas and a maximum size k, determine the value of
    the elementary symmetric polynomials:
    E(l+1,n+1) = sum_{J \subseteq 1..n,|J| = l} prod_{i \in J} lambda(i)
    :param lambda_value: the corresponding eigenvalues
    :param k: number of clusters
    :return:
    """
    N = lambda_value.shape[0]
    E = np.zeros((k + 1, N + 1))
    E[0, :] = 1

    for i in range(1, k+1):
        for j in range(1, N+1):
            E[i, j] = E[i, j - 1] + lambda_value[j-1] * E[i - 1, j - 1]

    return E

def sample_dpp_linear(X, num_batch, batch_size, max_epoch, data, axis = 1, print_step = -1, WITH_REPLACEMENT = False):
    import scipy.stats
    import scipy.linalg

    #assume data is csv of age and sex, is in order of generating the input matrix X
    age_sex = data[["age", "sex"]]
    age_sex['sex'].replace(['F','M'],[-1,1],inplace=True)
    age_sex=age_sex.apply(scipy.stats.zscore) #z-score meta-data

    #extract mete-data vectors to create kernel 
    age = age_sex['age']
    age = np.reshape([age],(len(age),1))
    age_transpose = np.transpose(age)

    sex = age_sex['sex']
    sex = np.reshape([sex],(len(sex),1))
    sex_transpose = np.transpose(sex)

    kern_original = np.dot(X.transpose(), X) #Linear covariance matrix based on imaging features

    idx_array_original = np.arange(start = 0, stop = X.shape[axis], step = 1)
    # idx_sample_full = -1 * np.ones(shape = (num_batch, batch_size, max_epoch)) #since indices are nonnegative, initialize indices to bogus value of negative one so that it is easy to check for mistake when debugging by checking if any value of idx_sample_full is negative
    idx_sample_full = -1 * np.ones(shape = (batch_size, num_batch, max_epoch)) #was originally in wrong shape

    #kernel does not change each epoch; no need to repeat this each epoch; run once and reuse evalue and evector
    evalue_original, evector_original = scipy.linalg.eigh(kern_original)

    if print_step < 0:
        print_step = max_epoch  #reset to print none to console

    for epoch in np.arange(max_epoch):

        if np.mod(epoch, print_step) == 0:
            print("%s: sample_dpp_gaussian: epoch: %i / %i - generating indices with DPP linear sampling for %i subjects." % (script_name, epoch, max_epoch, X.shape[axis]), flush=True)

        # idx_sample = -1 * np.ones((num_batch,batch_size))
        idx_sample = -1 * np.ones(shape = (batch_size, num_batch)) #was originally in wrong shape

        idx_array = np.copy(idx_array_original)
        kern = np.copy(kern_original)
        evector = np.copy(evector_original)
        evalue = np.copy(evalue_original)

        for sample in range(num_batch):
            if VERBOSE_FLAG:
                    print("%s: sample_dpp_linear: epoch: %i / %i | batch %i / %i - generating indices with sampling." % (script_name, epoch, max_epoch, sample, num_batch), flush=True)

            idx = sample_dpp( np.real(evalue),np.real(evector), batch_size)
            idx_sample[:, sample] = idx_array[idx]
            
            if not WITH_REPLACEMENT:
                kern = np.delete(kern, idx, 0)
                kern = np.delete(kern, idx, 1)
                idx_array = np.delete(idx_array, idx, 0)
                
                # need to also update evalue and evector
                evalue = np.delete(evalue, idx, 0)
                evector = np.delete(evector, idx, 0)
                evector = np.delete(evector, idx, 1)
        
        idx_sample_full[:,:,epoch] = idx_sample
        del idx_sample

    #sanity check
    if np.sum(idx_sample_full < 0):
        print("%s: sample_dpp_linear: error - there are at least %i indices where assignment of index failed to generate valid index greater than or equal to zero." % (script_name, np.sum(idx_sample_full < 0)), flush=True)
    return idx_sample_full

def sample_dpp_gaussian(X, num_batch, batch_size, max_epoch, data, sigma = 0.1, axis = 1, print_step = -1, WITH_REPLACEMENT = False):
    import scipy.stats
    age_sex = data[["age", "sex"]]
    age_sex['sex'].replace(['F','M'],[-1,1],inplace=True)
    age_sex=age_sex.apply(scipy.stats.zscore) #z-score meta-data

    #extract mete-data vectors to create kernel 
    age = age_sex['age']
    age = np.reshape([age],(len(age),1))
    age_transpose = np.transpose(age)

    sex = age_sex['sex']
    sex = np.reshape([sex],(len(sex),1))
    sex_transpose = np.transpose(sex)

    #kernel  
    #to-do: make into a function that takes some meta data (optional and required0 and generates a kernel
    kern_original = np.exp(-((age-age_transpose)**2+(sex-sex_transpose)**2)/sigma**2) #gaussian

    idx_array_original = np.arange(start = 0, stop = X.shape[axis], step = 1)
    # idx_sample_full = -1 * np.ones(shape = (num_batch, batch_size, max_epoch)) #since indices are nonnegative, initialize indices to bogus value of negative one so that it is easy to check for mistake when debugging by checking if any value of idx_sample_full is negative
    idx_sample_full = -1 * np.ones(shape = (batch_size, num_batch, max_epoch)) #was originally in wrong shape

    #kernel does not change each epoch; no need to repeat this each epoch; run once and reuse evalue and evector
    evalue_original, evector_original = scipy.linalg.eigh(kern_original)

    if print_step < 0:
        print_step = max_epoch  #reset to print none to console

    for epoch in np.arange(max_epoch):

        if np.mod(epoch, print_step) == 0:
            print("%s: sample_dpp_gaussian: epoch: %i / %i - generating indices with DPP gaussian sampling for %i subjects." % (script_name, epoch, max_epoch, X.shape[axis]), flush=True)

        idx_array = np.copy(idx_array_original)
        kern = np.copy(kern_original)
        evector = np.copy(evector_original)
        evalue = np.copy(evalue_original)

        # idx_sample = -1 * np.ones((num_batch,batch_size))
        idx_sample = -1 * np.ones(shape = (batch_size, num_batch)) #was originally in wrong shape

        for sample in range(num_batch):

            if VERBOSE_FLAG:
                print("%s: sample_dpp_gaussian: epoch: %i / %i | batch %i / %i - generating indices with sampling." % (script_name, epoch, max_epoch, sample, num_batch), flush=True)
            
            idx = sample_dpp(np.real(evalue),np.real(evector), batch_size)
            # idx_sample[sample,:] = idx_array[idx]
            idx_sample[:,sample] = idx_array[idx]
            
            if not WITH_REPLACEMENT:
                kern = np.delete(kern, idx, 0)
                kern = np.delete(kern, idx, 1)
                idx_array = np.delete(idx_array, idx, 0)

                # need to also update evalue and evector
                evalue = np.delete(evalue, idx, 0)
                evector = np.delete(evector, idx, 0)
                evector = np.delete(evector, idx, 1)

        idx_sample_full[:,:,epoch] = idx_sample
        del idx_sample

    #sanity check
    if np.sum(idx_sample_full < 0):
        print("%s: sample_dpp_linear: error - there are at least %i indices where assignment of index failed to generate valid index greater than or equal to zero." % (script_name, np.sum(idx_sample_full < 0)), flush=True)
    return idx_sample_full

def calculate_batch_loss_sum(X, idx_sample, W, SQUARED = True):
    """
    given indices of subjects for a specific batch, the final W from all the iterations/batches of an epoch, and X for original data to sample the indices from, calculate the sum of batch loss
    sum_{i=0}^{last batch}(|X_batch - W_final (W_final * X_batch)|_F^2)
    """
    batch_loss_sum = 0.0
    num_batch = idx_sample.shape[1]
    for batch_index in np.arange(num_batch):
        idx_sample_selected = (idx_sample[:, int(batch_index)]).astype(int)
        X_sampled = X[:, idx_sample_selected]
        del idx_sample_selected
                
        #for the sake of consistency, let us call the calculate_error function from utils module rather than calling frobenius norm function from scratch
        batch_loss_sum = batch_loss_sum + utils.calculate_error(X = X_sampled, W = W, H = np.matmul(W.T, X_sampled), SQUARED = SQUARED)
        del X_sampled
    return batch_loss_sum

## create argument parser with help documentation
parser=argparse.ArgumentParser(
  description = "This script is a mostly python conversion of opnmf_mem.m matlab script (https://github.com/asotiras/brainparts/blob/master/opnmf_mem.m) along with stochastic implementation on top of the conversion. It runs stochastic orthonormal projective non-negative matrix factorization (sopNMF). It has different stochastic sampling (uniform vs DPP) available for the stochastic sampling of subjects into mini-batches.",
  epilog = "Written by Abdalla Bani (bani.abdalla@wustl.edu) and Sung Min Ha (sungminha@wustl.edu)",
  add_help=True,
)
parser.add_argument("-i", "--inputFile",
type = str,
dest = "input_path",
required = False,
default = feret_path,
help = "Full path to the mat file containing variable X that contains input data to work with."
)
parser.add_argument("-d", "--demographic_data_path",
type = str,
dest = "demographic_data_path",
required = False,
default = csv_path,
help = "Full path to the csv file in the same order as the input X file of subject domain with relevant columns age and sex for sampling methods (such as dpp)."
)
parser.add_argument("-k", "--targetRank",
type = int,
dest = "target_rank",
required = False,
default = 40,
help = "What is the rank of the NMF you intend to run? This is the number of components that will be generated."
)
parser.add_argument("-m", "--maxEpoch",
type = int,
dest = "max_epoch",
required = False,
default = 5.0e4,
help = "Max number of iterations to optimize over. Note that if other stopping criteria are achived (e.g. tolerance), then the algorithm may stop before reaching this max number of iterations."
)
parser.add_argument("-t", "--tol",
type = float,
dest = "tol",
required = False,
default = 0.0e0,
help = "Tolerance value to use as threshold for stopping criterion. If the diffW = norm(W-W_old) / norm(W) is less than this tolerance value, then the iterations would stop for optimization regardless of whether max_epoch has been reached, under the assumption that the cost function has stabilized (plateau) and has reached close to local minimum."
)
parser.add_argument("-o", "--outputParentDir",
type = str,
dest = "output_parent_dir",
required = False,
default = os.path.join("/scratch/%s" % (username), "output_directory"),
help = "Path to output mat file that contain the outputs."
)
parser.add_argument("-0", "--initMeth",
type = str,
dest = "init_meth",
required = False,
default = "nndsvd",
help = "Method for initializing w0 for component. (random, nndsvd, nndsvda, nndsvdar)"
)
parser.add_argument("-u", "--updateMeth",
type = str,
dest = "update_meth",
required = False,
default = "mem",
help = "Method for update W: default is mem, and currently only mem is available. original (where XX^T is stored in memory) is not suitable for sopNMF where the mini-batch of X changes every epoch/iteration."
)
parser.add_argument("--multiplicativeUpdateMeth",
type = str,
dest = "multiplicative_update_meth",
required = False,
default = "normalize",
help = "Method for multiplicative update of W. (normalize, constant)"
)
parser.add_argument("-s", "--samplingMeth",
type = str,
dest = "sampling_meth",
required = False,
default = "uniform",
help = "Method for sampling of batch. (uniform, dppgaussian, dpplinear)"
)
parser.add_argument("-z", "--sampleSize",
type = int,
dest = "batch_size",
required = False,
default = 100,
help = "Number of subjects per mini-batch. Note that current implementation does not account for situation where you have a dividend after dividing total number of subjects (n, number of columns of X)"
)
parser.add_argument("--printStep",
type = int,
dest = "print_step",
required = False,
default = 1.0e1,
help = "Print progress every this many epochs."
)
parser.add_argument("--saveStep",
type = int,
dest = "save_step",
required = False,
default = 1.0e3,
help = "save progress (batch wise error per iteration, full data wise error, sparsity, elapsed time) every this many epochs into intermediate save files so that you can use those to track and plot changes over iterations/batches."
)
parser.add_argument("--restartStep",
type = int,
dest = "restart_step",
required = False,
default = 1.0e0,
help = "save progress every this many epochs. This will maintain a single file that gets overwritten this many epochs to serve a restart/checkpoint if the code fails."
)
parser.add_argument("--rho",
type = float,
dest = "rho",
required = False,
default = 0.25,
help = "For constant power multiplicative update."
)
parser.add_argument("--sigma",
type = float,
dest = "sigma",
required = False,
default = 0.1,
help = "For dpp gaussian kernel generation."
)
parser.add_argument("-V", "--verbose",
action = 'store_true',
dest = "VERBOSE_FLAG",
help = "Extra printouts for debugging."
)
parser.add_argument("-D", "--debug",
action = 'store_true',
dest = "DEBUG_FLAG",
help = "EXTRA EXTRA printouts for debugging."
)
parser.add_argument("--withReplacement",
action = 'store_true',
dest = "WITH_REPLACEMENT_FLAG",
help = "If set to true by calling this flag on, DPP sampling will happen with replacement instead of removing already selected sample"
)

#parse argparser
args = parser.parse_args()
input_path = args.input_path
demographic_data_path = args.demographic_data_path
target_rank = args.target_rank
tol = args.tol #tolerance
output_parent_dir = args.output_parent_dir
init_meth = args.init_meth
sampling_meth = args.sampling_meth
multiplicative_update_meth = args.multiplicative_update_meth
batch_size = args.batch_size
update_meth = args.update_meth
print_step = args.print_step
save_step = args.save_step
restart_step = args.restart_step
VERBOSE_FLAG = args.VERBOSE_FLAG
DEBUG_FLAG = args.DEBUG_FLAG    #also save all intermediates at save point
WITH_REPLACEMENT_FLAG = args.WITH_REPLACEMENT_FLAG
# iter_power = args.iter_power
max_epoch = int(args.max_epoch)  #same as number of iterations for opNMF, e.g.) 50K
rho = args.rho  #1/4 by default
sigma = args.sigma #0.1 by default
SQUARED_ERROR_FLAG = True
iteration_per_batch = int(1)
reset_small_value_threshold = 1.0e-16

utils.print_flush(string_variable = "input_path: ( %s )" % (input_path), script_name = script_name)
utils.exit_if_not_exist_file(input_path)
utils.print_flush(string_variable = "target_rank: ( %i )" % (target_rank), script_name = script_name)
utils.print_flush(string_variable = "tol: ( %0.5E )" % (tol), script_name = script_name)
utils.print_flush(string_variable = "output_parent_dir: ( %s )" % (output_parent_dir), script_name = script_name)
utils.print_flush(string_variable = "init_meth: ( %s )" % (init_meth), script_name = script_name)
utils.print_flush(string_variable = "sampling_meth: ( %s )" % (sampling_meth), script_name = script_name)
utils.print_flush(string_variable = "multiplicative_update_meth: ( %s )" % (multiplicative_update_meth), script_name = script_name)
utils.print_flush(string_variable = "batch_size: ( %i )" % (batch_size), script_name = script_name)
utils.print_flush(string_variable = "update_meth: ( %s )" % (update_meth), script_name = script_name)
utils.print_flush(string_variable = "print_step: ( %i )" % (print_step), script_name = script_name)
utils.print_flush(string_variable = "restart_step: ( %i )" % (restart_step), script_name = script_name)
utils.print_flush(string_variable = "save_step: ( %i )" % (save_step), script_name = script_name)
utils.print_flush(string_variable = "VERBOSE_FLAG: ( %r )" % (VERBOSE_FLAG), script_name = script_name)
utils.print_flush(string_variable = "DEBUG_FLAG: ( %r )" % (DEBUG_FLAG), script_name = script_name)
utils.print_flush(string_variable = "WITH_REPLACEMENT_FLAG: ( %r )" % (WITH_REPLACEMENT_FLAG), script_name = script_name)
utils.print_flush(string_variable = "max_epoch: ( %i )" % (max_epoch), script_name = script_name)
utils.print_flush(string_variable = "rho: ( %0.5E )" % (rho), script_name = script_name)
if sampling_meth == "dppgaussian":
    utils.print_flush(string_variable = "sigma: ( %0.5E )" % (sigma), script_name = script_name)

MEM_FLAG = False
if (update_meth == "mem"):
    pass
    MEM_FLAG = True
elif (update_meth == "original"):
    utils.print_flush("ERROR: sopNMF only supports update method of mem, not original.", script_name = script_name)
    sys.exit(1)
else:
    utils.print_flush("ERROR - Unknown update_meth (%s). Accepted values are mem and original. Exiting." % (update_meth), script_name = script_name)
    sys.exit(1)

## Variables Setup
outdir = os.path.join(output_parent_dir, "s-opNMF", "targetRank%i" % (target_rank), "init%s" % (init_meth), "update%s" % (update_meth), "tol%0.2E" % (tol), "maxEpoch%0.2E" % (max_epoch), "sampling%s" % (sampling_meth), "batchSize%i" % (batch_size))
if sampling_meth == "dppggaussian":
    outdir = os.path.join(outdir, "sigma%0.2E" % (sigma))
utils.print_flush(string_variable = "outdir: ( %s )" % (outdir), script_name = script_name)

if DEBUG_FLAG:
    VERBOSE_FLAG = True

#output naming scheme
output_basename_prefix = "sopNMF_%s" % (sampling_meth) 

#output path
output_path = os.path.join(outdir, "%s.mat" % (output_basename_prefix))
restart_path = os.path.join(outdir, "%s_restart.mat" % (output_basename_prefix))
restart_path_old = os.path.join(outdir, "%s_restart_old.mat" % (output_basename_prefix))
initialization_path = os.path.join(outdir, "%s_initialization.mat" % (output_basename_prefix))

#sanity check: does parent output directory exist?
utils.exit_if_not_exist_dir(dir_path = output_parent_dir, script_name = script_name)

#sanity check: does output exist?
if not os.path.isdir(outdir):
    os.makedirs(outdir)
utils.exit_if_exist_file(file_path = output_path, script_name = script_name)
utils.print_flush(string_variable = "outdir: ( %s )" % (outdir), script_name = script_name)

## initialize elapse time counts
total_elapsed_time = 0.0    #all time elapsed from this point on
initialization_elapsed_time = 0.0   #for initializations
saving_elapsed_time = 0.0   #for intermediate savings and loading from restart
statistics_elapsed_time = 0.0   #for calculating statistics (error/sparsity on the fly)
update_elapsed_time = 0.0   #for multiplicative update
sampling_elapsed_time = 0.0 #for sampling batches and selecting those batches as indices

total_start_time = time.time()
total_start_time_string = datetime.datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')
utils.print_flush("total start time\t: %s" % (total_start_time_string), script_name = script_name)

## Input Data Loading
saving_start_time = time.time()
utils.exit_if_not_exist_file(file_path = input_path, script_name = script_name)
utils.print_flush("Loading input variable X from ( %s )." % (input_path), script_name = script_name)
X = utils.load_hdf5storage_data(file_path = input_path, variable_name = "X")
m = np.shape(X)[0]
n = np.shape(X)[1]
utils.print_verbose("Loaded input variable X of shape [%i, %i] from ( %s )." % (m, n, input_path), script_name = script_name)

#load demographics data if not uniform sampling
if (sampling_meth == "dpplinear") or (sampling_meth == "dppgaussian"):
    utils.print_flush(string_variable = "demographic_data_path: ( %s )" %(demographic_data_path), script_name = script_name)
    utils.exit_if_not_exist_file(demographic_data_path)
    demographic_data = pd.read_csv(demographic_data_path)


#sanity check: does k <= n and k <= m
if target_rank > m:
    utils.print_flush("target_rank (%i) > m (%i, the number of features or rows of X). target_rank must be less than m. Exiting." % (target_rank, m), script_name = script_name)
    sys.exit(1)
if target_rank > n:
    utils.print_flush("target_rank (%i) > n (%i, the number of subjects or columns of X). target_rank must be less than n. Exiting." % (target_rank, n), script_name = script_name)
    sys.exit(1)
saving_elapsed_time = saving_elapsed_time + (time.time() - saving_start_time)
del saving_start_time

## load intermediate if it exists
if os.path.isfile(initialization_path):
    saving_start_time = time.time()
    utils.print_flush(string_variable = "Loading initialization from file ( %s )." % (initialization_path), script_name = script_name)
    w0 = utils.load_hdf5storage_data(file_path = initialization_path, variable_name = "w0", script_name = script_name)
    h0 = utils.load_hdf5storage_data(file_path = initialization_path, variable_name = "h0", script_name = script_name)
    total_elapsed_time = utils.load_hdf5storage_data(file_path = initialization_path, variable_name = "total_elapsed_time", script_name = script_name)
    saving_elapsed_time = utils.load_hdf5storage_data(file_path = initialization_path, variable_name = "saving_elapsed_time", script_name = script_name)
    initialization_elapsed_time = utils.load_hdf5storage_data(file_path = initialization_path, variable_name = "initialization_elapsed_time", script_name = script_name)
    num_batch = utils.load_hdf5storage_data(file_path = initialization_path, variable_name = "num_batch", script_name = script_name)
    idx_sample = utils.load_hdf5storage_data(file_path = initialization_path, variable_name = "idx_sample", script_name = script_name)
    saving_elapsed_time = saving_elapsed_time + (time.time() - saving_start_time)
    del saving_start_time

else:

    ## Initializations
    initialization_start_time = time.time()
    utils.print_verbose("Initializing W and H with %s intialization." % (init_meth), script_name = script_name)
    if (init_meth == "nndsvd") or (init_meth == "nndsvda") or (init_meth == "nndsvdar") or (init_meth == "random"):
        w0, h0 = initialize_nmf._initialize_nmf(X, target_rank, init=init_meth, eps=1e-6, random_state=None)
    else:
        utils.print_flush("ERROR - Unknown init_meth (%s). Exiting." % (init_meth), script_name = script_name)
        sys.exit(1)
    initialization_elapsed_time = initialization_elapsed_time + (time.time() - initialization_start_time)
    del initialization_start_time

    ## Sampling
    sampling_start_time = time.time()
    utils.print_verbose("Calculating num_batch based on n ( %i ) and batch_size ( %i )" % (n, batch_size), script_name = script_name)
    num_batch = int(np.ceil(np.divide(float(n), float(batch_size))))   #we need this value as int
    if (num_batch != np.divide(float(n), float(batch_size))):
        utils.print_flush("ERROR: currently, if your n (%i) divided by batch_size (%i) = (%f) does not end up as an integer (%i), then this code fails." % (n, batch_size, np.divide(float(n), float(batch_size)), num_batch), script_name = script_name)
    
    #generate indices for all iterations/epochs
    sampling_start_time = time.time()
    utils.print_flush("Sampling indices of subjects using %s sampling method" % (sampling_meth), script_name = script_name)
    if (sampling_meth == "uniform"):
        idx_sample = sample_uniform(X = X,  num_batch = int(num_batch), batch_size = int(batch_size), max_epoch = int(max_epoch), WITH_REPLACEMENT=WITH_REPLACEMENT_FLAG)
    elif (sampling_meth == "dpplinear"):
        idx_sample = sample_dpp_linear(X = X, num_batch = num_batch, batch_size = batch_size, max_epoch = max_epoch, data = demographic_data, axis = 1, print_step = int(np.round(max_epoch / 100.0)), WITH_REPLACEMENT=WITH_REPLACEMENT_FLAG )
    elif (sampling_meth == "dppgaussian"):
        idx_sample = sample_dpp_gaussian(X = X, num_batch = num_batch, batch_size = batch_size, max_epoch = max_epoch, data = demographic_data, axis = 1, print_step = int(np.round(max_epoch / 100.0)), sigma = sigma, WITH_REPLACEMENT=WITH_REPLACEMENT_FLAG )
    else:
        utils.print_flush("Unknown sampling method (%s). Exiting.", script_name = script_name)
        sys.exit(1)
    utils.print_flush("Finished sampling indices of subjects using %s sampling method" % (sampling_meth), script_name = script_name)
    sampling_elapsed_time = sampling_elapsed_time + (time.time() - sampling_start_time)
    del sampling_start_time


    #save initialization related variables for restarting
    total_elapsed_time = (time.time() - total_start_time) + total_elapsed_time
    saving_start_time = time.time()
    mdict = {
        "w0": w0,
        "h0": h0,
        "num_batch": num_batch,
        "saving_elapsed_time": saving_elapsed_time,
        "initialization_elapsed_time": initialization_elapsed_time,
        "total_elapsed_time": total_elapsed_time,
        "idx_sample": idx_sample
        }
    utils.print_flush("saving initialization file ( %s )." % (initialization_path), script_name = script_name)
    utils.save_intermediate_output(output_path = initialization_path, mdict = mdict)
    del mdict
    saving_elapsed_time = saving_elapsed_time + (time.time() - saving_start_time)
    del saving_start_time

initialization_start_time = time.time()
utils.print_verbose("Calculating XX^T to store in memory.", script_name = script_name)
if (MEM_FLAG == False):
    XX = np.matmul(X, np.transpose(X))

#initialize W to w0
W = w0
W_old = W

#initialize diffW for print purposes and for stopping criterion
diffW = 0.0

utils.print_verbose("X.shape = [%i, %i]" % (m, n)  , script_name = script_name)
utils.print_verbose("X.min max = [%0.15f, %0.15f]" % (np.amin(X, axis = None), np.amax(X, axis = None) )  , script_name = script_name)
utils.print_verbose("w0.shape = [%i, %i]" % (np.shape(w0)[0], np.shape(w0)[1]) , script_name = script_name)
utils.print_verbose("w0.min max = [%0.15f, %0.15f]" % (np.amin(w0, axis = None), np.amax(w0, axis = None) ) , script_name = script_name)
utils.print_verbose("h0.shape = [%i, %i]" % (np.shape(h0)[0], np.shape(h0)[1]) , script_name = script_name)
utils.print_verbose("h0.min max = [%0.15f, %0.15f]" % (np.amin(h0, axis = None), np.amax(h0, axis = None) )  , script_name = script_name)

#print information about batch sampling
utils.print_flush("batch_size: %i" % (batch_size))
utils.print_flush("num_batch: %i" % (num_batch))
utils.print_verbose("max_epoch: %i" % (max_epoch), script_name = script_name)
utils.print_verbose("batch_size (number of subjects per batch): %i" % (batch_size), script_name = script_name)
utils.print_verbose("num_batch (number of batches in full X): %i" % (num_batch), script_name = script_name)
utils.print_flush("iteration_per_batch: %i" % (iteration_per_batch), script_name = script_name)

#initalize arrays to store values during the for loop of updates
epoch_start = 0

epoch_array = np.arange(start = 0, stop = max_epoch, step = 1)
batch_loss_sum_per_epoch_array = np.zeros(shape = epoch_array.shape)
full_loss_per_epoch_array = np.zeros(shape = epoch_array.shape)
batch_loss_per_iteration_array = np.zeros(shape = (max_epoch, num_batch))
full_loss_per_iteration_array = np.zeros(shape = (max_epoch, num_batch))

sparsity_per_epoch_array = np.zeros(shape = epoch_array.shape)
sparsity_per_iteration_array = np.zeros(shape = (max_epoch, num_batch))

#measure time used for initializations
objective_function = 0.0    #update to value if VERBOSE_FLAG is on

#measure time
initialization_elapsed_time = initialization_elapsed_time + (time.time() - initialization_start_time)
del initialization_start_time

#load from save point if available
saving_start_time = time.time()
if os.path.isfile(restart_path):
    utils.print_flush(string_variable = "loading from save point ( %s )." % (restart_path), script_name = script_name)
    epoch_start = utils.load_intermediate_output(output_path = restart_path, variable_name = "epoch")
    batch_loss_sum_per_epoch_array = utils.load_intermediate_output(output_path = restart_path, variable_name = "batch_loss_sum_per_epoch_array")
    full_loss_per_epoch_array = utils.load_intermediate_output(output_path = restart_path, variable_name = "full_loss_per_epoch_array")
    batch_loss_per_iteration_array = utils.load_intermediate_output(output_path = restart_path, variable_name = "batch_loss_per_iteration_array")
    full_loss_per_iteration_array = utils.load_intermediate_output(output_path = restart_path, variable_name = "full_loss_per_iteration_array")
    sparsity_per_epoch_array = utils.load_intermediate_output(output_path = restart_path, variable_name = "sparsity_per_epoch_array")
    sparsity_per_iteration_array = utils.load_intermediate_output(output_path = restart_path, variable_name = "sparsity_per_iteration_array")
    W = utils.load_intermediate_output(output_path = restart_path, variable_name = "W")
    W_old = utils.load_intermediate_output(output_path = restart_path, variable_name = "W_old")
    diffW = utils.load_intermediate_output(output_path = restart_path, variable_name = "diffW")
    saving_elapsed_time = utils.load_intermediate_output(output_path = restart_path, variable_name = "saving_elapsed_time") + saving_elapsed_time   #in addition to current sum
    update_elapsed_time = utils.load_intermediate_output(output_path = restart_path, variable_name = "update_elapsed_time") + update_elapsed_time   #in addition to current sum
    sampling_elapsed_time = utils.load_intermediate_output(output_path = restart_path, variable_name = "sampling_elapsed_time") + sampling_elapsed_time
    statistics_elapsed_time = utils.load_intermediate_output(output_path = restart_path, variable_name = "statistics_elapsed_time") + statistics_elapsed_time
    initialization_elapsed_time = utils.load_intermediate_output(output_path = restart_path, variable_name = "initialization_elapsed_time") + initialization_elapsed_time
    total_elapsed_time = utils.load_intermediate_output(output_path = restart_path, variable_name = "total_elapsed_time") + total_elapsed_time

    utils.print_flush(string_variable = "epoch restart point: ( %i )" % (epoch_start), script_name = script_name)
    utils.print_flush(string_variable = "saving_elapsed_time (s): ( %f )" % (saving_elapsed_time), script_name = script_name)
    utils.print_flush(string_variable = "update_elapsed_time (s): ( %f )" % (update_elapsed_time), script_name = script_name)
    utils.print_flush(string_variable = "sampling_elapsed_time (s): ( %f )" % (sampling_elapsed_time), script_name = script_name)
    utils.print_flush(string_variable = "statistics_elapsed_time (s): ( %f )" % (statistics_elapsed_time), script_name = script_name)
    utils.print_flush(string_variable = "initialization_elapsed_time (s): ( %f )" % (initialization_elapsed_time), script_name = script_name)
    utils.print_flush(string_variable = "total_elapsed_time (s): ( %f )" % (total_elapsed_time), script_name = script_name)
saving_elapsed_time = saving_elapsed_time + (time.time() - saving_start_time)
del saving_start_time

#start for loop
for epoch in np.arange(start = epoch_start, stop = epoch_array[-1], step = 1):
    
    #save restart point
    saving_start_time = time.time()
    if np.mod(epoch, restart_step) == 0:
        if epoch != 0:
            #to avoid filesystem io error (e.g. syncing across nodes or some other activity that may cause slight slowdown/lag and cause the code to fail to locate the file to move)
            time.sleep(0.5) #sleep for 0.5 seconds
            shutil.move(src = restart_path, dst = restart_path_old)
            time.sleep(0.5) #sleep for 0.5 seconds

        utils.print_verbose(string_variable = "epoch %i / %i: saving intermediate save point for restarting." % (epoch, max_epoch), script_name = script_name)
        mdict = {
            "epoch": epoch,
            "batch_loss_sum_per_epoch_array": batch_loss_sum_per_epoch_array, 
            "full_loss_per_epoch_array": full_loss_per_epoch_array,
            "batch_loss_per_iteration_array": batch_loss_per_iteration_array,
            "full_loss_per_iteration_array": full_loss_per_iteration_array,
            "sparsity_per_epoch_array": sparsity_per_epoch_array,
            "sparsity_per_iteration_array": sparsity_per_iteration_array,
            "W": W,
            "W_old": W_old, 
            "diffW": diffW,
            "saving_elapsed_time": saving_elapsed_time,
            "update_elapsed_time": update_elapsed_time,
            "sampling_elapsed_time": sampling_elapsed_time,
            "statistics_elapsed_time": statistics_elapsed_time, 
            "initialization_elapsed_time": initialization_elapsed_time,
            "total_elapsed_time": total_elapsed_time + (time.time() - total_start_time)
        }
        utils.save_intermediate_output(output_path = restart_path, mdict = mdict, OVERWRITE = True)
        if epoch == 0:
            utils.save_intermediate_output(output_path = restart_path_old, mdict = mdict, OVERWRITE = False)
        del mdict
            
    saving_elapsed_time = saving_elapsed_time + (time.time() - saving_start_time)
    del saving_start_time

    for batch_index in np.arange(num_batch):    #e.g. 40 num batch, 1 iteration per sample, 50K max_epoch, which means there will be 25 samples per batch
        
        ## Batch Sampling: select the batch of subjects from full X for this iteration
        sampling_start_time = time.time()
        utils.print_debug(string_variable = "epoch %i / %i: selecting %i-th batch with %i subjects." % (epoch, max_epoch, batch_index, batch_size), script_name = script_name)
        idx_sample_selected = (idx_sample[:, int(batch_index), int(epoch)]).astype(int)
        X_sampled = X[:, idx_sample_selected]
        del idx_sample_selected
        sampling_elapsed_time = sampling_elapsed_time + (time.time() - sampling_start_time)
        del sampling_start_time

        for iteration in np.arange(iteration_per_batch):  #one iteration per batch

            #print progress to console
            if np.mod(epoch, print_step) == 0 and batch_index == 0:
                current_elapsed_time = (time.time() - total_start_time) + (total_elapsed_time)
                remaining_time_string = utils.get_remaining_time_string(iteration = epoch, max_iter = max_epoch, elapsed_time = (time.time() - total_start_time) + total_elapsed_time )
                utils.print_flush(string_variable = "epoch: %i / %i \t| batch: %i / %i \t| iteration: %i / %i \t| diffW = %0.5E \t| elapsed time: %f seconds\t| remaining time (approx): %s" % (epoch, max_epoch, batch_index, num_batch, iteration, iteration_per_batch, diffW, current_elapsed_time, remaining_time_string), script_name = script_name)
                del current_elapsed_time, remaining_time_string

            update_start_time = time.time()
            utils.print_debug(string_variable = "Storing old W", script_name = script_name)
            W_old = W

            #multiplicative update for W
            utils.print_debug(string_variable = "Update with %s multiplicative update rule on W" % (multiplicative_update_meth), script_name = script_name)
            if multiplicative_update_meth == "normalize":
                W = opnmf_update_rule.multiplicative_update(X = X_sampled, W = W, OPNMF = True, MEM = MEM_FLAG)
            elif multiplicative_update_meth == "constant":
                W = opnmf_update_rule.multiplicative_update(X = X_sampled, W = W, OPNMF = True, MEM = MEM_FLAG, rho = rho)
            else:
                print("%s: multiplicative_update_meth (%s) must be either normalize or constant. Exiting." % (script_name, multiplicative_update_meth), flush=True)
                sys.exit(1)

            #reset small values of W - to prevent slowdown of computations caused by very small values, call on reset_small_value function to reset element of W with values smaller than (default 1.0e-16) to (default 1.0e-16)
            utils.print_debug(string_variable = "Resetting small values of W", script_name = script_name)
            W = opnmf_update_rule.reset_small_value(W, min_reset_value = reset_small_value_threshold)

            #normalize W following mulitplicative update to stabilize W; only if not using normalization mulitplicative update version, not for convergent (constant root)
            if multiplicative_update_meth == "normalize":
                utils.print_debug(string_variable = "Normalizing W", script_name = script_name)
                W = opnmf_update_rule.normalize_W(W = W)

            ## stopping criterion: diffW for normalization or convergent update using diffW
            diffW = np.linalg.norm(W_old - W, ord = 'fro') / np.linalg.norm(W_old, ord = 'fro')
            utils.print_debug(string_variable = "diffW = %0.5E" % (diffW), script_name = script_name)
            if multiplicative_update_meth == "constant" or multiplicative_update_meth == "normalize":
                if diffW < tol:
                    utils.print_flush("Converged after %i / %i epochs, %i / %i batches, %i / %i iterations." % (epoch, max_epoch, batch_index, num_batch, iteration, iteration_per_batch), script_name = script_name)
                    
            update_elapsed_time = update_elapsed_time + (time.time() - update_start_time)
            del update_start_time

            ## calculate statistics per batch level
            statistics_start_time = time.time()
            if iteration == iteration_per_batch - 1:
                utils.print_debug(string_variable = "calculating iteration-level (batch) level loss and sparsity", script_name = script_name)

                # save loss (Batch-wise and Full Data-wise) at iteration(batch) level
                batch_loss_per_iteration_array[int(epoch), int(batch_index)] = utils.calculate_error(X = X_sampled, W = W, H = np.matmul(W.T, X_sampled), SQUARED = SQUARED_ERROR_FLAG) # |X_batch - W_batch * (W_batch' * X_batch)|_F^2
                full_loss_per_iteration_array[int(epoch), int(batch_index)] = utils.calculate_error(X = X, W = W, H = np.matmul(W.T, X), SQUARED = SQUARED_ERROR_FLAG)  # |X_full - W_batch * (W_batch' * X_full)|_F^2

                # save sparsity at iteration level
                sparsity_per_iteration_array[int(epoch), int(batch_index)] = utils.calculate_sparsity(W = W)
            statistics_elapsed_time = time.time() - statistics_start_time
            del statistics_start_time

            ## save epoch level statistics (including batch level saved metrics)
            saving_start_time = time.time()
            if iteration == iteration_per_batch - 1:
                ## save at the end of every epoch the batch/full loss and sparsity per iteration(batch)
                if np.mod(epoch, save_step) == 0:
                    # if batch_index == num_batch - 1:
                    if batch_index == 0:
                        output_intermediate_path = os.path.join(outdir, "%s_data_epoch%05d.mat" % (output_basename_prefix, epoch))
                        mdict = {
                            "batch_loss_per_iteration_array": batch_loss_per_iteration_array[int(epoch), :],
                            "full_loss_per_iteration_array": full_loss_per_iteration_array[int(epoch), :],
                            "sparsity_per_iteration_array": sparsity_per_iteration_array[int(epoch), :],
                            "saving_elapsed_time": saving_elapsed_time,
                            "update_elapsed_time": update_elapsed_time,
                            "sampling_elapsed_time": sampling_elapsed_time,
                            "statistics_elapsed_time": statistics_elapsed_time, 
                            "initialization_elapsed_time": initialization_elapsed_time,
                            "total_elapsed_time": total_elapsed_time + (time.time() - total_start_time),
                            "W": W,
                            "elapsed_time": total_elapsed_time + (time.time() - total_start_time)
                            }
                        utils.save_intermediate_output(output_path = output_intermediate_path, mdict = mdict)
                        del output_intermediate_path, mdict
            saving_elapsed_time = saving_elapsed_time + (time.time() - saving_start_time)
            del saving_start_time

    #approximate the error (loss) of final W for the epoch in estimating EACH of the sampled X, to be used as stopping criterion
    saving_start_time = time.time()
    batch_loss_sum_per_epoch_array[int(epoch)] = calculate_batch_loss_sum(X = X, idx_sample = idx_sample[:, :, int(epoch)], W = W, SQUARED = SQUARED_ERROR_FLAG)
    full_loss_per_epoch_array[int(epoch)] = utils.calculate_error(X = X, W = W, H = np.matmul(W.T, X), SQUARED = SQUARED_ERROR_FLAG)
    sparsity_per_epoch_array[int(epoch)] = utils.calculate_sparsity(W = W)
    saving_elapsed_time = saving_elapsed_time + (time.time() - saving_start_time)
    del saving_start_time

## Calculate H
utils.print_flush(string_variable = "calculating final H", script_name = script_name)
H = np.matmul(np.transpose(W), X)

#save final output
saving_start_time = time.time()
utils.print_flush(string_variable = "saving final output (%s)" % (output_path), script_name = script_name)
mdict = {
    "X": X,
    "W": W,
    "H": H,
    "batch_loss_per_iteration_array": batch_loss_per_iteration_array,
    "full_loss_per_iteration_array": full_loss_per_iteration_array,
    "sparsity_per_iteration_array": sparsity_per_iteration_array,
    "batch_loss_sum_per_epoch_array": batch_loss_sum_per_epoch_array,
    "full_loss_per_epoch_array": full_loss_per_epoch_array,
    "sparsity_per_epoch_array": sparsity_per_epoch_array,
    "total_elapsed_time": total_elapsed_time,
    "initialization_elapsed_time": initialization_elapsed_time,
    "saving_elapsed_time": saving_elapsed_time,
    "statistics_elapsed_time": statistics_elapsed_time,
    "update_elapsed_time": update_elapsed_time
}
utils.save_intermediate_output(output_path = output_path, mdict = mdict)
del mdict
saving_elapsed_time = saving_elapsed_time + (time.time() - saving_start_time)
del saving_start_time

total_end_time = time.time()
total_end_time_string = datetime.datetime.fromtimestamp(total_end_time).strftime('%Y-%m-%d %H:%M:%S')
utils.print_flush(string_variable = "start time\t: %s" % (total_start_time_string), script_name = script_name)
utils.print_flush(string_variable = "end time\t: %s" % (total_end_time_string), script_name = script_name)
del total_start_time_string, total_end_time_string
total_elapsed_time = total_elapsed_time + (total_end_time - total_start_time)
utils.print_flush(string_variable = "Total optimization elapsed time: %f seconds" % (total_elapsed_time))
utils.print_flush(string_variable = "Saving/Restarting/IO: %f seconds" % (saving_elapsed_time))
utils.print_flush(string_variable = "Multiplicative Updates: %f seconds" % (update_elapsed_time))
utils.print_flush(string_variable = "Initialization: %f seconds" % (initialization_elapsed_time))
utils.print_flush(string_variable = "Statistics Calculations: %f seconds" % (statistics_elapsed_time))