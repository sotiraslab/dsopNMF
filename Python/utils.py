"""
Python snippets and functions to help facilitate saving and loading volumetric and surface data to be used with NMF.
"""
import os #for checking if filepaths are reasonable, and files/directories exist
if (os.name == 'nt'):
    from asyncio.windows_events import NULL
import sys #check python version, exit upon sanity check failures
import numpy as np #for doing statistics and matrix manipulation
import pandas as pd #for reading and saving csv / list files and managing lists in general
import nibabel as nib #for reading and saving neuroimaging files
import h5py #for saving as hdf5 mat file; prerequisite to hdf5storage
import hdf5storage #for saving as hdf5 mat file (v7.3 on matlab); has much better compression than scipy savemat
import time #for calculating remaining time
import datetime #for calculating remaining time

#default value for script name to be used for printing to console
script_name=os.path.basename(__file__)
module_name=os.path.basename(__file__)

##### OS and System Related Functions
def check_python_version(major=3, minor=6):
    """
    using sys module, check if currently used python version is above the user provided version (3.6 default)
    """
    if sys.version_info[0] < major or sys.version_info[1] < minor:
        print("ERROR: This script requires python version %i.%i. You are using version %i.%i. Exiting." % (major, minor, sys.version_info[0], sys.version_info[1]), flush = True)
        sys.exit(1)

def exit_if_not_exist_dir(dir_path, script_name = script_name):
    """
    Using os module to check whether a directory exists, and exit using sys module if it does not exist.
    """
    if not os.path.isdir(dir_path):
        print("%s: ERROR - directory ( %s ) does not exist. Exiting." % (script_name, dir_path), flush=True)
        sys.exit(1)

def exit_if_not_exist_file(file_path, script_name = script_name):
    """
    Using os module to check whether a file exists, and exit using sys module if it does not exist.
    """
    if not os.path.isfile(file_path):
        print("%s: ERROR - file ( %s ) does not exist. Exiting." % (script_name, file_path), flush=True)
        sys.exit(1)

def exit_if_exist_dir(dir_path, script_name = script_name):
    """
    Using os module to check whether a directory exists, and exit using sys module if it does exist.
    """
    if os.path.isdir(dir_path):
        print("%s: ERROR - directory ( %s ) already exists. Exiting." % (script_name, dir_path), flush=True)
        sys.exit(1)

def exit_if_exist_file(file_path, script_name = script_name):
    """
    Using os module to check whether a file exists, and exit using sys module if it does exist.
    """
    if os.path.isfile(file_path):
        print("%s: ERROR - file ( %s ) already exists. Exiting." % (script_name, file_path), flush=True)
        sys.exit(1)

def print_flush(string_variable, script_name=script_name ):
    if isinstance(string_variable, str):
        print("%s: %s" % (script_name, string_variable), flush=True)
    else:
        print("%s: " % (script_name))
        print(string_variable, flush=True)

def print_verbose(string_variable, VERBOSE_FLAG = False, script_name=script_name ):
    if VERBOSE_FLAG:
        if isinstance(string_variable, str):
            print("%s: VERBOSE - %s" % (script_name, string_variable), flush=True)
        else:
            print("%s: VERBOSE" % (script_name))
            print(string_variable, flush=True)

def print_debug(string_variable, DEBUG_FLAG = False, script_name=script_name ):
    if DEBUG_FLAG:
        if isinstance(string_variable, str):
            print("%s: DEBUG - %s" % (script_name, string_variable), flush=True)
        else:
            print("%s: DEBUG" % (script_name))
            print(string_variable, flush=True)

##### Image Processing Related Functions
def mask_image(
    image,  #type = np.array
    mask,   #type = np.array
    VERBOSE=False
):
    """
    provided an image and mask, return a masked array with background removed
    """

    #check if image and mask are same size
    if (np.shape(image) != np.shape(mask)):
        print("\nERROR: image and mask do not share the same shape.", flush=True)
        print("\timage.shape =\n", flush=True)
        print(np.shape(image), flush=True)
        print("\tmask.shape =\n", flush=True)
        print(np.shape(mask), flush=True)
        print("\n\tExiting.", flush=True)
        sys.exit(1)

    #apply mask as overlay
    masked = np.copy(image)
    masked[mask==0] = 0 #set where mask is background as zero, where mask is nonzero as is from original image
    
    #return masked image
    return masked

def calculate_mask_image(
    data_matrix,        #m by n matrix
    w,                  #x-direction size of each image (w * h = m where data_matrix is m x n) 
    h,                  #y-direction size of each image (w * h = m where data_matrix is m x n) 
    threshold = 0.0,    #assuming after z-scoring
    VERBOSE = False
):
    """
    z score the images, then create a mask for values greater than some threshold in that z scored image
    """
    import scipy.stats
    
    #data shape sanity check
    X_shape = np.shape(data_matrix)
    m = X_shape[0]
    n = X_shape[1]
    if len(X_shape) != 2:
        print("ERROR: expected data matrix to be 2-D, not %i-D. Exiting." % (len(X_shape)), flush=True)
        sys.exit(1)
    if m != w * h:
        print("ERROR: expected (w * h = %i * %i) which equals %i to be equal to (m = %i). Exiting." % (w, h, w*h, m), flush=True)
        sys.exit(1)
    
    #calculate average
    X_zscored = scipy.stats.zscore(data_matrix, axis = 1)

    if (VERBOSE):
        print("VERBOSE: X_zscored.shape = \t", flush=True)
        print(np.shape(X_zscored), flush=True)
        print("VERBOSE: X_zscored.min.max = [%f, %f]" % (np.amin(X_zscored), np.amax(X_zscored)), flush=True)

    #initialize mask
    mask = np.zeros(shape = np.shape(X_zscored))
    mask[X_zscored > threshold] = 1 #only values where X_zscored is greater than threshold is set to foreground (ROI)

    if (VERBOSE):
        print("VERBOSE: mask.shape = \t", flush=True)
        print(np.shape(mask), flush=True)
        print("VERBOSE: mask.min.max = [%f, %f]" % (np.amin(mask), np.amax(mask)), flush=True)

    return mask

def load_nibabel_data(
    csv_path,           #type: str
    file_type = "nii",  #type: str
    mask_path = "None",      #type: str
    VERBOSE = False,     #type: bool
    list_header = None
):
    """
    read in list of nii.gz / nii or mgh files of n scans each with m voxels from the provided list. Then create a matrix X of size m by n and return the matrix. This assumes that the list file has no header line, and only one column with full paths to nii.gz files

    inputs:
        csv_path: full path to csv file with a single column with full paths to nii/nii.gz/mgh files to read from. The csv file should not contain header
        file_type: valid values are nii and mgh, which decides how the file is being read by nibabel
        VERBOSE: print additional lines to help debugging and to track progress of the function

    outputs:
        X: data matrix of size m by n
        list_df: pandas dataframe with full paths, with the files that were missing from input list removed

    To-Do:
        masking out regions based on some roi provided
    """

    #check if list file exists
    if not os.path.isfile(csv_path):
        print("ERROR: list file ( %s ) does not exist. Exiting." % (csv_path), flush=True)
        sys.exit(1)

    #check if valid value was given for file_type
    if file_type == "nii":
        if VERBOSE:
            print("VERBOSE: file_type = ( %s ): Using nibabel load function to read nifti files." % (file_type), flush=True)
    elif file_type == "mgh":
        if VERBOSE:
            print("VERBOSE: file_type = ( %s ): Using nibabel freesurfer io function to read mgh files." % (file_type), flush=True)
    else:
        print("ERROR: unknown file_type = ( %s ). Valid values are nii or mgh. Exiting." % (file_type), flush=True)
        return

    #read csv file
    list_df = pd.read_csv(csv_path, header=list_header)
    list_df.column = ["fullpath"]

    #check how many are listed
    n_list = np.shape(list_df)[0]

    #check how many have valid files to read from (remove the ones that do not have valid, existing files to read from)
    n = 0
    for index, row in list_df.iterrows():
        if (VERBOSE):
            print("VERBOSE: %i / %i - checking if file ( %s ) exists." % (index, n_list), flush=True)
        file_path = row["fullpath"]

        if os.path.isfile(file_path):
            n = n + 1
        else:
            print("ERROR: file ( %s ) does not exist. Skipping this file." % (file_path), flush=True)
        
    print("%i / %i files are available to be read and used as input data." % (n, n_list), flush=True)

    #get list of indices with missing files to drop from list_df
    indices_to_remove = -1 * np.ones(shape = (n_list-n, ))
    indices_to_remove_index = 0
    for index, row in list_df.iterrows():
        file_path = row["fullpath"]
        if not os.path.isfile(file_path):
            indices_to_remove[indices_to_remove_index] = index
            indices_to_remove_index = indices_to_remove_index + 1
    del indices_to_remove_index
    del file_path

    #drop rows from list_df
    list_df.drop(labels = indices_to_remove, inplace = True, axis = 0 ) #axis = 0 means drop from index not column
    del indices_to_remove

    #read sample image to get dimensions of single subject
    if file_type == "nii":
        file_path = list_df["fullpath"][0]
        sample = nib.load(file_path).get_fdata()
        sample_shape = np.shape(sample)
        m = 1
        for dimension in np.arange(np.shape(sample_shape)[0]): #num dimension
            m = m * sample_shape[dimension]
        del dimension
        del sample_shape
        del file_path
        del sample
    elif file_type == "mgh":
        file_path = list_df["fullpath"][0]
        sample = nib.freesurfer.mghformat.load(file_path).get_fdata()
        m = np.shape(sample)[0]
        del file_path
        del sample

    #load mask if available
    if file_type == "nii":
        if os.path.isfile(mask_path): #assumes only for nii files not mgh
            mask = nib.load(mask_path).get_fdata()        

    #initialize X with correct size    
    print("Generating data matrix X of shape [ %i, %i ]. " % (m, n))
    X = np.zeros(shape = (m,n))

    #in case index of data frame is different from the actual index with which data matrix X is generated
    list_df["subject_index"] = list_df.index

    count = 0
    for index, row in list_df.iterrows():
        file_path = row["fullpath"]
        list_df["subject_index"] = count

        if file_type == "nii":
            sample = nib.load(file_path).get_fdata()

            #mask if available
            if os.path.isfile(mask_path):
                sample = mask_image(sample, mask)
        elif file_type == "mgh":
            sample = nib.freesurfer.mghformat.load(file_path).get_fdata()
        
        print("X[:, %i] = ( %s )" % (count, file_path), flush=True)
        X[:, count] = np.reshape(sample, newshape = (m, ))
        count = count + 1

    return X, list_df

def load_hdf5storage_data(
    file_path,           #type: str
    VERBOSE = False,     #type: bool
    variable_name = "X",
    script_name = script_name
):
    """
    read in hdf5storage (matlab compatible version 7.3) mat file
    
    inputs:
        file_path: full path to mat file
        VERBOSE: print additional lines to help debugging and to track progress of the function
        variable_name: variable name to extract from file_path

    outputs:
        X: data matrix of size m by n
    """

    #check if list file exists
    if not os.path.isfile(file_path):
        print("%s: %s: ERROR: mat file ( %s ) does not exist. Exiting." % (script_name, module_name, file_path), flush=True)
        sys.exit(1)
    
    #load data
    if VERBOSE:
        print("%s: %s: loading mat file ( %s )." % (script_name, module_name, file_path), flush=True)
    X = hdf5storage.loadmat(file_name = file_path, variable_names=[variable_name])[variable_name]
    return X

def calculate_error(X, W, H, SQUARED = False):
    #2022-09-27: changed from frobenius norm of (X-WH) to frobenius norm squared
    error = np.linalg.norm(X - np.matmul(W, H), ord='fro')
    if SQUARED:
        error = np.power(error, 2)
    return error

def calculate_sparsity(W):
    D = np.shape(W)[0]
    numerator = np.sum( np.abs(W), axis=0)
    denominator = np.sqrt(np.sum(np.power(W,2),axis=0))
    subtract_by = np.divide(numerator, denominator)
    del numerator, denominator

    subtract_from = np.sqrt(D)
    subtracted = subtract_from - subtract_by
    del subtract_from, subtract_by

    numerator = np.mean(subtracted)
    del subtracted

    denominator = np.sqrt(D)-1
    sparsity = np.divide( numerator, denominator)
    return sparsity

def calculate_relative_value(ref_value, new_value):
    difference = new_value - ref_value
    divided_by = np.divide(difference, ref_value)
    percentage = np.multiply(divided_by, 100)
    return percentage

def clustering_adjustedRand_fast(u, v):
    """
    python port of Aris's clustering_adjustedRand_fast.m
    """
    m = np.maximum(np.max(u, axis = 0), np.max(v, axis=0))

    va = np.zeros(shape = (m, ) )
    vb = np.zeros(shape = (m, ) )
    mat = np.zeros(shape = (m, m))

    print_debug("clustering_adjustedRand_fast: m (%i)" % (m))
    print_debug("clustering_adjustedRand_fast: va.shape, vb.shape, mat.shape")
    print_debug(va.shape)
    print_debug(vb.shape)
    print_debug(mat.shape)

    for i in np.arange(m):
        va[i] = np.sum(u == i)
        vb[i] = np.sum(v == i)
        hu = (u == i)
        for j in np.arange(m):
            hv = ( v == j)
            mat[i, j] = np.sum( np.multiply(hu, hv))

    ra = np.divide( np.sum( np.sum( np.multiply( mat, np.transpose(mat - np.ones(shape = (m,m))) ), axis = 0), axis = 0), 2.0)

    rb = np.divide( np.matmul(va, np.transpose(va - np.ones(shape = (1, m))) ), 2.0)
    
    rc = np.divide( np.matmul(vb, np.transpose(vb - np.ones(shape = (1, m))) ), 2.0)

    rn = ( np.shape(u)[0] * (np.shape(u)[0] - 1) ) / 2.0

    r = (ra-rb*rc/rn) /( 0.5*rb+0.5*rc-rb*rc/rn )
    return r

def get_binarized_mask(mask, mask_threshold = 0.8):
    """
    given a 3D data matrix, create a binary mask where the voxels of data matrix that are greater than the mask_threshold is set to 1 and rest are set to 0.

    inputs:
        mask: 3D data matrix
        mask_threshold = (0.8 by default)
    """
    binarized_mask = np.zeros(shape = mask.shape)
    binarized_mask[mask > mask_threshold] = 1.0
    return binarized_mask

def get_union_mask(mask1, mask2, mask1_threshold = 0.8, mask2_threshold = 0.8, VERBOSE = False):
    """
    take two 3D data matrices of same size, and create a union mask where either of teh two matrices are greater than some threshold value. Note that this is a binary mask with values {0, 1}

    inputs:
        mask1: 3D data matrix
        mask2: 3D data matrix
        mask1_threshold = (0.8 by default)
        mask2_threshold = (0.8 by default)
    """
    #check if two are same shape
    if mask1.shape != mask2.shape:
        print("%s: shape of mask1 is not the same as mask2" % (script_name), flush=True)
        print("%s: mask1.shape = " % (script_name))
        print(mask1.shape)
        print("%s: mask2.shape = " % (script_name))
        print(mask2.shape)
        print("%s: Exiting function.", flush=True)
        return
    
    mask1_binarized = get_binarized_mask(mask1, mask_threshold = mask1_threshold)
    if VERBOSE:
        print("%s: VERBOSE: number of voxels in mask1 > 0:" % (script_name))
        print(np.sum(mask1_binarized > 0), flush=True)

    mask2_binarized = get_binarized_mask(mask2, mask_threshold = mask2_threshold)
    if VERBOSE:
        print("%s: VERBOSE: number of voxels in mask2 > 0:" % (script_name))
        print(np.sum(mask2_binarized > 0), flush=True)

    union_mask_binarized = np.zeros(shape = mask1.shape)
    union_mask_binarized[mask1_binarized > 0] = 1.0
    union_mask_binarized[mask2_binarized > 0] = 1.0
    return union_mask_binarized

def isequal_header(nii1, nii2, VERBOSE = False):
    """
    compare q and s forms from two nii structures. If they are equal, return true, and return false if they are different.
    
    inputs:
        nii1: nibabel nifti structure 1
        nii2: nibabel nifti structure 2
    """
    header1 = nii1.header
    header2 = nii2.header

    qform1 = header1.get_qform()
    qform2 = header2.get_qform()

    sform1 = header1.get_sform()
    sform2 = header2.get_sform()

    shape1 = header1.get_data_shape()
    shape2 = header2.get_data_shape()

    voxel_dim1 = header1.get_zooms()
    voxel_dim2 = header2.get_zooms()
    
    iequal_header_boolean = True
    if not np.array_equal(qform1, qform2):
        if VERBOSE:
            print("%s: isequal_header: qform is different." % (script_name), flush=True)
        iequal_header_boolean = False

    if not np.array_equal(sform1, sform2):
        if VERBOSE:
            print("%s: isequal_header: sform is different." % (script_name), flush=True)
        iequal_header_boolean = False

    if not np.array_equal(shape1, shape2):
        if VERBOSE:
            print("%s: isequal_header: image shape is different." % (script_name), flush=True)
        iequal_header_boolean = False

    if not np.array_equal(voxel_dim1, voxel_dim2):
        if VERBOSE:
            print("%s: isequal_header: voxel dimension is different." % (script_name), flush=True)
        iequal_header_boolean = False

    return iequal_header_boolean

def save_intermediate_output(output_path, mdict, OVERWRITE = False, VERBOSE = False):
    """
    save W to some output path
    """

    #if output does not exist, save W
    if not OVERWRITE:
        if os.path.isfile(output_path):
            print("%s: output_path ( %s ) already exists." % (script_name, output_path), flush=True)
            return

    if VERBOSE:
        print("%s: VERBOSE: saving output ( %s )." % (script_name, output_path), flush=True)
    hdf5storage.savemat(file_name = output_path, mdict = mdict)
    return

def load_intermediate_output(output_path, variable_name, VERBOSE = False):
    """
    load specific variable for intermediate output path
    """

    #if output does not exist, save W
    if not os.path.isfile(output_path):
        print("%s: output_path ( %s ) does not exist." % (script_name, output_path), flush=True)
        return

    if VERBOSE:
        print("%s: VERBOSE: loading variable ( %s ) from output ( %s )." % (script_name, variable_name, output_path), flush=True)
    data = hdf5storage.loadmat(file_name = output_path, variable_names = [variable_name])[variable_name]
    return data

def get_remaining_time_string(iteration, max_iter, elapsed_time, VERBOSE = False):
    """
    get a projected remaining time based on current number of iterations elapsed & total iterations
    """
    if VERBOSE:
        if iteration <= 0:
            projected_total_time = datetime.timedelta(seconds = 1)
        else:
            projected_total_time = datetime.timedelta( seconds = (max_iter / iteration) * elapsed_time)
        projected_total_time_days = int(projected_total_time.days)
        projected_total_time_hours = int(np.floor_divide(projected_total_time.seconds, 60 * 60))
        projected_total_time_minutes = int(np.floor_divide(projected_total_time.seconds - projected_total_time_hours * 60 * 60, 60))
        projected_total_time_seconds = int(projected_total_time.seconds - projected_total_time_hours * 60 * 60 - projected_total_time_minutes * 60)
        projected_total_time_string = "%i-%02d:%02d:%02d" % (projected_total_time_days, projected_total_time_hours, projected_total_time_minutes, projected_total_time_seconds)
        print("%s: VERBOSE: projected total time: %s" % (script_name, projected_total_time_string), flush=True)

    #using datetime.datetime is too much hassle
    # if (iteration <= 0):
    #     projected_remaining_time = 0
    # else:
    #     projected_remaining_time = ( (max_iter - iteration) / iteration) * elapsed_time #in seconds
    # projected_remaining_time_days = int(np.floor_divide(projected_remaining_time, 60 * 60 * 24))
    # projected_remaining_time_hours = int(np.floor_divide(projected_remaining_time, 60 * 60) - (projected_remaining_time_days) * 24)
    # projected_remaining_time_minutes = int(np.floor_divide(projected_remaining_time, 60) - ( (projected_remaining_time_days * 24 + projected_remaining_time_hours) * 60 ))
    # projected_remaining_time_seconds = int(projected_remaining_time - ( (projected_remaining_time_days * 24 * 60 + projected_remaining_time_hours * 60 + projected_remaining_time_minutes) * 60))
    
    #using datetime.timedelta
    if (iteration <= 0):
        projected_remaining_time = datetime.timedelta(seconds = 1)
    else:
        projected_remaining_time = datetime.timedelta(seconds =(( (max_iter - iteration) / iteration) * elapsed_time ) )

    projected_remaining_time_days = int(projected_remaining_time.days)
    projected_remaining_time_hours = int(np.floor_divide(projected_remaining_time.seconds, 60 * 60))
    projected_remaining_time_minutes = int(np.floor_divide(projected_remaining_time.seconds - projected_remaining_time_hours * 60 * 60, 60))
    projected_remaining_time_seconds = int(projected_remaining_time.seconds - projected_remaining_time_hours * 60 * 60 - projected_remaining_time_minutes * 60)
    projected_remaining_time_string = "%i-%02d:%02d:%02d" % (projected_remaining_time_days, projected_remaining_time_hours, projected_remaining_time_minutes, projected_remaining_time_seconds)
    return projected_remaining_time_string

##### Sampling (for stochastic opNMF)
def sample_uniform(X,  num_sample, sample_size, axis = 1, random_number_seed = 0):
    """
    inputs
    X: input matrix of size [m,n]
    axis (default=1): which direction to sample from; default is n in X in [m,n] where n is assumed to be subjects
    """
    # rng = np.random.seed(4)   #old method, numpy does not recommend this as this sets GLOBAL variable
    # To-Do: sanity checks; is X 2D?
    # To-Do: sanity checks; does num_sample * sample_size <= m * n (where X is [m, n])
    # To-Do: sanity checks; are all values of num_sample * sample_size batches unique?

    idx_list = np.arange(start = 0, stop = np.shape(X)[axis], step = 1)
    rng = np.random.default_rng(random_number_seed)
    idx_sample = rng.choice(idx_list, size = (num_sample, sample_size), replace = False)
    return idx_sample

# dpp sampling: new attempt by Sung
def sample_dpp(X,  num_sample, sample_size, var1, var2, axis = 1, random_number_seed = 0):
    """
    attempt to do DPP with same input output structure as sample_uniform function; assume var1 and var2 are raw values (e.g. age and sex)
    """
    #import scipy.linalg needed for eigenvalue/eigenvector calculation
    import scipy.linalg
    #import scipy.stats needed for z-scoring
    import scipy.stats

    idx_list = np.arange(start = 0, stop = np.shape(X)[axis], step = 1) #construct a list of indices to refer to subjects in X
    rng = np.random.default_rng(random_number_seed)

    #get age (or other equivalent first variable) from raw var1 after z-scoring
    age_zscored = scipy.stats.zscore(var1)
    age_zscored_transpose = np.transpose(age_zscored)

    #get sex (or other equivalent second variable) from raw var2 after z-scoring
    sex_zscored = scipy.stats.zscore(var2)
    sex_zscored_transpose = np.transpose(sex_zscored)
    
    #sigma: standard deviation; is this correct?
    sigma = 1   #placeholder
    #iintialize kernel
    kern = np.exp(-((age_zscored-age_zscored_transpose)**2+(sex_zscored-sex_zscored_transpose)**2)/sigma**2) #gaussian

    #get eigenvector and eigenvalue
    evalue, evector = scipy.linalg.eigh(kern)

    #k = num_sample
    #v = sample_size

    V = evector[:, sample_size]

# dpp sampling: old reference
def sample_dpp_reference(evalue, evector, k):
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


def get_objective_function(X, W, trXtX = -1, OPNMF = False):
    """
    ||X-WH||_F^2, where ||*||_F is frobenius norm, and in our case of oPNMF/pNMF, H = W^TX
    Hence, ||X-WW^TX||_F^2

    We note that for opNMF with H=W'X and W'W = I, the frobenius norm |X-WW'tX|_F^2 = |X|_F^2 - |W'X|_F^2
    We can calculate |X|_F^2 prior to for loop and reuse the value instead of calculating it every update

    For pNMF iwht H=W'X, we don't have orthogonality constraint W'W=I, so the simplification is not as far and one can only simplify frobenius norm above to the following:

    """
    FULL = False
    if trXtX == -1:
        FULL = True
        OPNMF = False

    if FULL == True:
        obj = np.power(np.linalg.norm( X - np.matmul(W, np.matmul(W.T, X)) , 'fro') , 2)
    else:
        WtX = np.matmul(W.T, X)

        if OPNMF:
            obj = trXtX - np.power(np.linalg.norm(WtX, 'fro'), 2)
        else:
            obj = trXtX - 2 * np.power(np.linalg.norm(WtX, 'fro'), 2) + np.power(np.linalg.norm(np.matmul(W,WtX), 'fro'), 2)
    return obj