"""
opNMF implemented in python

Current implementation has following options:
    opNMF (regular version with XX^T stored in memory)
    opNMF (mem version with reduced memory footprint but longer runtime)
    qr-opNMF
    svd-opNMF
    tsvd-opNMF

"""

import sys #check python version, exit upon sanity check failures
import os #path checks
import getpass  #for getting username on compute jobs since os.getlogin() works in interactive jobs but not compute jobs

import numpy as np #matrix multiplications and optimizations
import argparse #for taking in user input/parameter/switch
import time #for measuring elapsed time
import datetime #for default output path string formation
import hdf5storage #for saving as hdf5 mat file (v7.3 on matlab); has much better compression than scipy savemat

import utils #for loading and saving data
import initialize_nmf #for W component initialization
import opnmf_update_rule #for W component initialization

#default value for script name to be used for printing to console
script_name = os.path.basename(__file__)
script_dir = os.path.dirname(__file__)
username=str(getpass.getuser())

def save_datapoint(output_parent_dir, output_basename_prefix, iteration, loss, sparsity, elapsed_time, VERBOSE_FLAG = False):
    """
    for plotting purpose, not for restarting from checkpoint
    """

    #define output path to save intermediate file to
    output_path = os.path.join(output_parent_dir, "%s_plottingData_iteration%06d.mat" % (output_basename_prefix, iteration))

    #ensure output path does not exist
    if os.path.isfile(output_path):
        print("%s: save_intermediate_savepoint: output path ( %s ) already exist. Exiting." % (script_name,  output_path), flush=True)
        sys.exit(1)

    mdict = {
        "loss": loss,
        "sparsity": sparsity,
        "elapsed_time": elapsed_time,
        "iteration": iteration
    }

    #if verbose, print out to console where you are saving datapoint to
    if (VERBOSE_FLAG == True):
        print("%s: save_datapoint: saving loss, sparsity, and elapsed time at iteration ( %s ) to file ( %s )." % (script_name, iteration, output_path), flush=True)
    hdf5storage.savemat(output_path, mdict = mdict)
    del mdict
    del output_path
    return

def save_intermediate_savepoint(output_parent_dir, output_basename_prefix, epoch, sample, W, iteration):
    """
    to work with load_intermediate_savepoint for restarting from a checkpoint
    """
    
    #define output path to save intermediate file to
    output_path = os.path.join(output_parent_dir, "%s_intermediate_epoch%05d-batch%05d-totalIteration%05d.mat" % (output_basename_prefix, epoch, sample, iteration))

    #ensure output path does not exist
    if os.path.isfile(output_path):
        print("%s: save_intermediate_savepoint: output path ( %s ) already exist. Exiting." % (script_name,  output_path), flush=True)
        sys.exit(1)
    
    #generate dictionary of objects to save
    mdict = {
        "W": W,
        "epoch": epoch,
    }

    #save file
    hdf5storage.savemat(file_name = output_path, mdict = mdict)

#sanity check: python version
utils.check_python_version(major = 3, minor = 7) #ensure the user is using python3.7 or higher

#default paths
git_dir = os.path.join("/scratch/sungminha/git/NMF_Testing")
csv_path = os.path.join(git_dir, "output_directory/20220314_OASIS_Surface_opNMF/oasis3_freesurfer_id_only_onlyBaseline_withMRString_fwhm20_fsaverage_lh_thickness_fullpath.csv")
feret_path = os.path.join(git_dir, "faces_1024_2409_min0_max1_double.mat")
EPSILON = np.finfo(np.float32).eps

## create argument parser with help documentation
parser=argparse.ArgumentParser(
  description = "This script is a mostly python conversion of opnmf.m matlab script. It runs orthonormal projective non-negative matrix factorization. In addition, it has options to allow SVD or QR decomposition on input data matrix to optimize update rules.",
  epilog = "Written by Sung Min Ha (sungminha@wustl.edu)",
  add_help=True,
)
parser.add_argument("-i", "--inputFile",
type = str,
dest = "input_path",
required = False,
default = feret_path,
help = "Full path to the mat file containing variable X that contains input data to work with or a csv file containing list of nii.gz or mgh files to read and construct into input matrix."
)
parser.add_argument("-k", "--targetRank",
type = int,
dest = "target_rank",
required = False,
default = 20,
help = "What is the rank of the NMF you intend to run? This is the number of components that will be generated."
)
parser.add_argument("-m", "--maxIter",
type = int,
dest = "max_iter",
required = False,
default = 1.0e4,
help = "Max number of iterations to optimize over. Note that if other stopping criteria are achived (e.g. tolerance), then the algorithm may stop before reaching this max number of iterations."
)
parser.add_argument("-t", "--tol",
type = float,
dest = "tol",
required = False,
default = 1.0e-6,
help = "Tolerance value to use as threshold for stopping criterion. If the diffW = norm(W-W_old) / norm(W) is less than this tolerance value, then the iterations would stop for optimization regardless of whether max_iter has been reached, under the assumption that the cost function has stabilized (plateau) and has reached close to local minimum."
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
help = "Method for update W. (svd, qr, mem, randpca, truncsvd, original)"
)
parser.add_argument("-p", "--printStep",
type = int,
dest = "print_step",
required = False,
default = 1.0e3,
help = "Print progress every this many steps."
)
parser.add_argument("-s", "--saveStep",
type = int,
dest = "save_step",
required = False,
default = 2.0e3,
help = "save progress every this many steps."
)
parser.add_argument("--restartStep",
type = int,
dest = "restart_step",
required = False,
default = 1.0e2,
help = "save restart check point every this many steps."
)
parser.add_argument("-c", "--calculateStep",
type = int,
dest = "calculate_step",
required = False,
default = 1.0e2,
help = "calculate objective function |X-WH|_F every this many steps."
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
parser.add_argument("--svdRank",
type = int,
dest = "svd_rank",
default = 2.0e6,    #2^6 = 64
help = "If you choose svd update method (where you do SVD(XX^T) to simplify update rule, how many ranks do you want to keep from SVD(XX^T)?"
)
parser.add_argument("--qrRank",
type = int,
dest = "qr_rank",
default = 2.0e6,    #2^6 = 64
help = "If you choose qr update method (where you do qr(X) to simplify update rule, how many ranks do you want to keep from qr(X)?"
)
parser.add_argument("--n_oversamples",
type = int,
dest = "n_oversamples",
default = 10,    #2^6 = 64
help = "If you choose randpca or truncsvd update methods (where you do randpca(X) or truncsvd(X) to simplify update rule), how many n_iter (power iterations) do you want?"
)
parser.add_argument("--n_iter",
type = int,
dest = "n_iter",
default = 4,    #2^6 = 64
help = "If you choose randpca or truncsvd update methods (where you do randpca(X) or truncsvd(X) to simplify update rule), how many n_oversamples (additional number of oversampled vectors) do you want?"
)
parser.add_argument("--saveEveryIteration",
action = 'store_true',
dest = "SAVE_EVERY_EPOCH_FLAG",
help = "Save every iteration(epoch) the value of error/sparsity/elapsed_time. Note of warning that this will take up more space in storage."
)
parser.add_argument("--frobeniusNormErrorSquared",
action = 'store_true',
dest = "FROBENIUS_NORM_ERROR_SQUARED",
help = "Instead of |X-WH|_F for loss, use |X-WH|_F^2 for loss with square."
)


#parse argparser
args = parser.parse_args()
input_path = args.input_path
target_rank = int(args.target_rank)
max_iter = int(args.max_iter) #maximum number of iterations
tol = args.tol #tolerance
output_parent_dir = args.output_parent_dir
init_meth = args.init_meth
update_meth = args.update_meth
print_step = int(args.print_step)
save_step = int(args.save_step)
restart_step = int(args.restart_step)
calculate_step = int(args.calculate_step)
VERBOSE_FLAG = args.VERBOSE_FLAG
SAVE_EVERY_EPOCH_FLAG = args.SAVE_EVERY_EPOCH_FLAG
DEBUG_FLAG = args.DEBUG_FLAG    #also save all intermediates at save point
qr_rank = int(args.qr_rank)      #only if using qr update_meth
svd_rank = int(args.svd_rank)    #only if using svd update_meth
n_oversamples =int(args.n_oversamples)  #only if using truncsvd or randpca
n_iter = int(args.n_iter)
FROBENIUS_NORM_ERROR_SQUARED = args.FROBENIUS_NORM_ERROR_SQUARED

## Variables Setup
outdir = os.path.join(output_parent_dir, "opNMF", "targetRank%i" % (target_rank), "init%s" % (init_meth), "update%s" % (update_meth), "tol%0.2E" % (tol), "maxIter%0.2E" % (max_iter))
if (update_meth == "svd") or (update_meth == "truncsvd"):
    outdir = os.path.join(outdir, "svdRank%i" % (svd_rank))
elif (update_meth == "randpca"):
    outdir = os.path.join(outdir, "pcaRank%i" % (svd_rank))
elif (update_meth == "qr"):
    outdir = os.path.join(outdir, "qrRank%i" % (qr_rank))

if DEBUG_FLAG:
    VERBOSE_FLAG = True

#output naming scheme
output_basename_prefix = "opNMF_%s" % (update_meth)

#output path
output_path = os.path.join(outdir, "%s.mat" % (output_basename_prefix))

#print inputs from argparser
utils.print_flush(string_variable = "input_path: ( %s )" % (input_path), script_name = script_name)
utils.print_flush(string_variable = "output_parent_dir: ( %s )" % (output_parent_dir), script_name = script_name)
utils.print_flush(string_variable = "outdir: ( %s )" % (outdir), script_name = script_name)
utils.print_flush(string_variable = "output_path: ( %s )" % (output_path), script_name = script_name)
utils.print_flush(string_variable = "target_rank: ( %i )" % (target_rank), script_name = script_name)
utils.print_flush(string_variable = "init_meth: ( %s )" % (init_meth), script_name = script_name)
utils.print_flush(string_variable = "update_meth: ( %s )" % (update_meth), script_name = script_name)
utils.print_flush(string_variable = "max_iter: ( %i )" % (max_iter), script_name = script_name)
utils.print_flush(string_variable = "tol: ( %0.5E )" % (tol), script_name = script_name)
utils.print_flush(string_variable = "print_step: ( %i )" % (print_step), script_name = script_name)
utils.print_flush(string_variable = "save_step: ( %i )" % (save_step), script_name = script_name)
utils.print_flush(string_variable = "restart_step: ( %i )" % (restart_step), script_name = script_name)
if ( update_meth == "svd" ):
    utils.print_flush(string_variable = "svd_rank: ( %i )" % (svd_rank), script_name = script_name)
if ( update_meth == "qr" ):
    utils.print_flush(string_variable = "qr_rank: ( %i )" % (qr_rank), script_name = script_name)
if (update_meth == "randpca"):
    utils.print_flush(string_variable = "svd_rank: ( %i )" % (svd_rank), script_name = script_name)
if (update_meth == "truncsvd"):
    utils.print_flush(string_variable = "svd_rank: ( %i )" % (svd_rank), script_name = script_name)
if (update_meth == "randpca") or (update_meth == "truncsvd"):
    utils.print_flush(string_variable = "n_oversamples: ( %i )" % (n_oversamples), script_name = script_name)
    utils.print_flush(string_variable = "n_iter: ( %i )" % (n_iter), script_name = script_name)
if (update_meth == "normalize"):
    utils.print_flush(string_variable = "update_meth: ( %s ) - treated as mem" % (update_meth), script_name = script_name)

utils.print_flush(string_variable = "FROBENIUS_NORM_ERROR_SQUARED: ( %r )" % (FROBENIUS_NORM_ERROR_SQUARED), script_name = script_name)

utils.exit_if_not_exist_dir(dir_path = output_parent_dir, script_name = script_name)
#sanity check: does output exist?
if not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok = True)
utils.exit_if_exist_file(file_path = output_path, script_name = script_name)

utils.exit_if_not_exist_file(file_path = input_path, script_name = script_name)
X = utils.load_hdf5storage_data(file_path = input_path, variable_name = "X")

initialization_start_time = time.time()

if VERBOSE_FLAG:
    utils.print_verbose("X has shape [%i, %i]." % (np.shape(X)[0], np.shape(X)[1]), script_name=script_name)

if (update_meth == "original"):
    XX = np.matmul(X, np.transpose(X))
    utils.print_flush("update method ( %s ): XX has shape = [ %i by %i ]" % (update_meth, XX.shape[0], XX.shape[1]), script_name = script_name)

#initialize components and coefficients W0 and H0
if (init_meth == "nndsvd") or (init_meth == "nndsvda") or (init_meth == "nndsvdar") or (init_meth == "random"):
    w0, h0 = initialize_nmf._initialize_nmf(X, target_rank, init=init_meth, eps=1e-6, random_state=None)
else:
    utils.print_flush("ERROR - Unknown init_meth (%s). Exiting." % (init_meth), script_name = script_name)
    sys.exit(1)

if (update_meth == "qr"):
    Q, R = opnmf_update_rule.qr(X, rank = qr_rank)
elif (update_meth == "svd"):
    U, s, Vh = opnmf_update_rule.svd(X, rank = svd_rank)
    S = s * np.eye(N = np.shape(s)[0])
    US = np.matmul(U, S)
elif (update_meth == "truncsvd"):
    US, s, Vh = opnmf_update_rule.truncated_svd(X, rank = svd_rank, n_oversamples = n_oversamples, n_iter = n_iter )
    S = s * np.eye(N = np.shape(s)[0])
elif (update_meth == "randpca"):
    U, Vh, randpca_mean = opnmf_update_rule.randomized_pca(X, rank = svd_rank, n_oversamples = n_oversamples, n_iter = n_iter)
elif (update_meth == "mem") or (update_meth == "normalize"):
    pass
elif (update_meth == "original"):
    pass
else:
    utils.print_flush("ERROR - Unknown update_meth (%s). Exiting." % (update_meth), script_name = script_name)
    sys.exit(1)

initialization_end_time = time.time()
initialization_elapsed_time = initialization_end_time - initialization_start_time
del initialization_start_time
del initialization_end_time

#initialize W to w0
W = w0
W_old = W    

#initialize diffW for print purposes and for stopping criterion
diffW = 0.0

loss_array = np.zeros(shape = (max_iter, ))
sparsity_array = np.zeros(shape = (max_iter, ))
elapsed_time_array = np.zeros(shape = (max_iter, ))

#save initialization related variables for restarting
initialization_path = os.path.join(outdir, "%s_initialization.mat" % (output_basename_prefix))
mdict = {"w0": w0, "h0": h0, "initialization_elapsed_time": initialization_elapsed_time}
if (update_meth == "qr"):
    mdict_temp = {"Q":Q, "R":R}
    mdict.update(mdict_temp)
    del mdict_temp
elif (update_meth == "svd"):
    mdict_temp = {"U":U, "s":s, "Vh":Vh}
    mdict.update(mdict_temp)
    del mdict_temp
elif (update_meth == "truncsvd"):
    mdict_temp = {"US":US, "Vh":Vh}
    mdict.update(mdict_temp)
    del mdict_temp
elif (update_meth == "randpca"):
    mdict_temp = {"U":U, "Vh":Vh, "randpca_mean": randpca_mean}
    mdict.update(mdict_temp)
    del mdict_temp
utils.save_intermediate_output(output_path = initialization_path, mdict = mdict)
del mdict

total_time = initialization_elapsed_time    #total time includes initialization + iterations
start_time = time.time()
start_time_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
utils.print_flush("start time\t: %s" % (start_time_string), script_name = script_name)
objective_function = 0.0    #update to value if VERBOSE_FLAG is on

iteration_restart = 0
#load from intermediate file if available
intermediate_path = os.path.join(outdir, "%s_intermediate.mat" % (output_basename_prefix))
if os.path.isfile(intermediate_path) and os.path.isfile(initialization_path):
    total_time = utils.load_intermediate_output(output_path = intermediate_path, variable_name = "total_time")
    diffW = utils.load_intermediate_output(output_path = intermediate_path, variable_name = "diffW")
    W = utils.load_intermediate_output(output_path = intermediate_path, variable_name = "W")
    W_old = utils.load_intermediate_output(output_path = intermediate_path, variable_name = "W_old")
    iteration_restart = utils.load_intermediate_output(output_path = intermediate_path, variable_name = "iteration")
    
    w0 = utils.load_intermediate_output(output_path = initialization_path, variable_name = "w0")
    h0 = utils.load_intermediate_output(output_path = initialization_path, variable_name = "h0")
    if (update_meth == "qr"):
        Q = utils.load_intermediate_output(output_path = initialization_path, variable_name = "Q")
        R = utils.load_intermediate_output(output_path = initialization_path, variable_name = "R")
    elif (update_meth == "svd"):
        U = utils.load_intermediate_output(output_path = initialization_path, variable_name = "U")
        s = utils.load_intermediate_output(output_path = initialization_path, variable_name = "s")
        Vh = utils.load_intermediate_output(output_path = initialization_path, variable_name = "Vh")
        S = s * np.eye(N = np.shape(s)[0])
        US = np.matmul(U, S)
    elif (update_meth == "truncsvd"):
        US = utils.load_intermediate_output(output_path = initialization_path, variable_name = "US")
        # s = utils.load_intermediate_output(output_path = initialization_path, variable_name = "s")
        Vh = utils.load_intermediate_output(output_path = initialization_path, variable_name = "Vh")
        # S = s * np.eye(N = np.shape(s)[0])
    elif (update_meth == "randpca"):
        U = utils.load_intermediate_output(output_path = initialization_path, variable_name = "U")
        Vh = utils.load_intermediate_output(output_path = initialization_path, variable_name = "Vh")
        randpca_mean = utils.load_intermediate_output(output_path = initialization_path, variable_name = "randpca_mean")

for iteration in np.arange(start = iteration_restart, stop = max_iter, step = 1):

    if np.mod(iteration, print_step) == 0:
        current_time = time.time()
        current_time_string = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
        elapsed_time = current_time - start_time
        remaining_time_string = utils.get_remaining_time_string(iteration = iteration, max_iter = max_iter, elapsed_time = elapsed_time + total_time)
        if VERBOSE_FLAG:
            objective_function = np.linalg.norm(X - np.matmul(W, np.matmul(np.transpose(W), X)), ord = "fro")
        utils.print_flush( string_variable = "iteration: %i / %i\t| diffW = %0.5E\t| current time: %s\t| elapsed time: %f seconds\t| remaining time (approx): %s\t| objective function = %0.5E" % (iteration, max_iter, diffW, current_time_string, elapsed_time, remaining_time_string, objective_function), script_name = script_name)
        
    if np.mod(iteration, restart_step) == 0 and not SAVE_EVERY_EPOCH_FLAG:
        elapsed_time = time.time() - start_time
        utils.print_verbose(string_variable = "saving intermediate savepoint (%s)" % (intermediate_path), script_name = script_name)
        mdict = {
            "total_time": total_time + elapsed_time,
            "W": W,
            "W_old": W_old,
            "iteration" : iteration,
            "diffW": diffW
        }
        hdf5storage.savemat(file_name = intermediate_path, mdict = mdict)
        del mdict

    if SAVE_EVERY_EPOCH_FLAG:
        intermediate_debug_path = os.path.join(outdir, "%s_intermediate_epoch%05d.mat" % (output_basename_prefix, iteration))
        current_time = time.time()
        elapsed_time = current_time - start_time
        del current_time
        error = utils.calculate_error(X = X, W = W, H = (np.matmul(W.T, X)), SQUARED = FROBENIUS_NORM_ERROR_SQUARED)
        sparsity = utils.calculate_sparsity(W = W)
        mdict = {
            "iteration": iteration, 
            "error": error,
            "sparsity": sparsity,
            "elapsed_time": elapsed_time}
        utils.save_intermediate_output(output_path = intermediate_debug_path, mdict = mdict)
        del error, sparsity, elapsed_time
    elif (np.mod(iteration, save_step) == 0):
        intermediate_debug_path = os.path.join(outdir, "%s_intermediate_epoch%05d.mat" % (output_basename_prefix, iteration))
        current_time = time.time()
        elapsed_time = current_time - start_time
        del current_time
        #unless debug flag is on, only save the latest
        mdict = {
            "W": W,
            "iteration" : iteration,
            "elapsed_time": total_time + elapsed_time
        }   #for visualization purpose
        utils.save_intermediate_output(output_path = intermediate_debug_path, mdict = mdict)
        del mdict

    #update error, sparsity, and elapsed_time
    if SAVE_EVERY_EPOCH_FLAG:
        loss = utils.calculate_error(X = X, W = W, H = np.matmul(W.T, X), SQUARED = FROBENIUS_NORM_ERROR_SQUARED)
        loss_array[iteration] = loss
        sparsity = utils.calculate_sparsity(W = W)
        sparsity_array[iteration] = sparsity
        elapsed_time = time.time() - start_time
        elapsed_time_array[iteration] = elapsed_time

        #save data for plotting
        save_datapoint(output_parent_dir = outdir, output_basename_prefix = output_basename_prefix, loss = loss, sparsity = sparsity, elapsed_time = elapsed_time, iteration = iteration)
        del loss, sparsity, elapsed_time

    #update W_old for old W value
    W_old = W
    REJECT_W = False

    if (update_meth == "qr"):
        W = opnmf_update_rule.multiplicative_update_qr(W = W, Q = Q, R = R, script_name = script_name, SANITY_CHECK_FLAG = False)
    elif (update_meth == "svd"):
        W = opnmf_update_rule.multiplicative_update_svd(W = W, US = US, script_name = script_name, SANITY_CHECK_FLAG = False)
    elif (update_meth == "truncsvd"):
            W = opnmf_update_rule.multiplicative_update_svd(W = W, US = US, script_name = script_name, SANITY_CHECK_FLAG = False)
    elif (update_meth == "randpca"):
        W = opnmf_update_rule.multiplicative_update_randpca(U = U, Vh = Vh, mean = randpca_mean, script_name = script_name, SANITY_CHECK_FLAG = False)
    elif (update_meth == "mem") or (update_meth == "normalize"):
        W = opnmf_update_rule.multiplicative_update(X = X, W = W, OPNMF = True, MEM = True, rho = 1, DEBUG = False, script_name = script_name)
    elif (update_meth == "original"):
        W = opnmf_update_rule.multiplicative_update(W = W, XX = XX, OPNMF = True, MEM = False, rho = 1, script_name = script_name, DEBUG = False)
    else:
        utils.print_flush("ERROR - unknown update method (%s). Exiting." % (update_meth), script_name = script_name)
        sys.exit(1)

    # MATLAB: As the iterations were progressing, computational time per iteration was increasing due to operations involving really small values
    utils.print_debug("resettting minimum value of W to avoid computational cost.")
    W = opnmf_update_rule.reset_small_value(W = W, min_reset_value = 1e-16)

    #normalize
    if (update_meth == "normalize") or (update_meth == "original"):
        utils.print_debug(string_variable = "normalizing W.", script_name = script_name)
        W = opnmf_update_rule.normalize_W(W = W)

    #stopping criterion
    if not REJECT_W:
        utils.print_debug(string_variable = "updating diffW.", script_name = script_name)
        diffW = np.linalg.norm(W_old - W, ord = 'fro') / np.linalg.norm(W_old, ord = 'fro')
        if diffW < tol:
            utils.print_flush("Converged after %i steps." % (iteration))
            break

end_time = time.time()
end_time_string = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
utils.print_flush(string_variable = "start time\t: %s" % (start_time_string), script_name = script_name)
utils.print_flush(string_variable = "end time\t: %s" % (end_time_string), script_name = script_name)
elapsed_time = end_time - start_time
total_time = total_time + elapsed_time  #total_time is for adding up all time across multiple save/restart from checkpoints, elapsed time is only for current run since last restart
utils.print_flush(string_variable = "Optimization loop for W elapsed time: %f seconds (for current job, not full number of iterations if this job was restarted from save point)" % (elapsed_time))

## Calculate H
utils.print_flush(string_variable = "calculating final H", script_name = script_name)
H = np.matmul(np.transpose(W), X)

#save final output
utils.print_flush(string_variable = "saving final output (%s)" % (output_path), script_name = script_name)
mdict = {"X": X, "W": W, "H": H, "w0": w0, "h0": h0, "elapsed_time": total_time, "initialization_elapsed_time":initialization_elapsed_time, "target_rank": target_rank, "iteration": iteration, "tol": tol, "max_iter": max_iter, "init_meth": init_meth}
utils.save_intermediate_output(output_path = output_path, mdict = mdict)
del mdict