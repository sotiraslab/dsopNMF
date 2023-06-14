"""
code snippet to initialize the component W and H with, using basic QR and SVD to break down XX^T.
"""
import os
import sys
import numpy as np
import scipy.linalg
import scipy.stats

script_name=os.path.basename(__file__)
module_name=os.path.basename(__file__)

def get_objective_function(X, W, trXtX = -1, W_ORTHOGONAL = False):
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

        if W_ORTHOGONAL:   #WARNING: this will only work if W'W = I is true at all iterations; for most opNMF, this assumption is not really true at least at earlier/initial iterations; I could not get this simplified reconstruction error to work with adaptive update for simple opNMF
            obj = trXtX - np.power(np.linalg.norm(WtX, 'fro'), 2)
        else:
            obj = trXtX - 2 * np.power(np.linalg.norm(WtX, 'fro'), 2) + np.power(np.linalg.norm(np.matmul(W,WtX), 'fro'), 2)
    return obj

def normalize_W(W):
    """
    normalize W by 2-norm of itself and return the normalized W
    """
    W = np.divide( W , np.linalg.norm(W, ord = 2) ) #matching matlab (2-norm) for normalization

    return W

def multiplicative_update(W, X = -1, OPNMF = True, MEM = True, rho = 1, XX = -1, DEBUG = False, EPSILON = np.finfo(np.float32).eps, script_name = script_name):
    """
    pnmf matlab mem: W = W .* (2X*(X'*W)) ./ (W*((W'*X)*(X'*W)) + X*(X'*W)*(W'*W));
    pnmf matlab original: W = W .* (2XX*W) ./ (W*(W'*XX*W) + XX*W*(W'*W));

    opnmf matlab mem: W = W .* (X*(X'*W)) ./ (W*((W'*X)*(X'*W)));
    opnmf matlab original: W = W .* (XX*W) ./ (W*(W'*XX*W));

    after experimenting, we noted that pNMF required the 2 in the numerator only if stabilization by normalization is not performed; if no normalization is performed and one performs update using convergence guaranteed update rule with cubic or fourth root, then leaving the 2 out will screw up the reconstruction error of pNMF
    
    ##### original update rule from https://ieeexplore.ieee.org/document/5438836
    ##### new update rule to avoid normalization from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6045343&tag=1; guarantees monotonically decreasing function

    added DEBUG flag so that you can decide to slow down each iteration at the cost of doing a sanity check for dimension mismatch
    """
    if DEBUG:
        #if DEBUG mode is on, do extra sanity checks per iteration
        m = np.shape(X)[0]
        n = np.shape(X)[1]
        k = np.shape(W)[1]

        #sanity checks: dimension mismatches
        if np.shape(W)[0] != m:
            print("%s: X has shape [%i, %i], but W has shape [%i, %i]. W should have shape [%i, %i]. Exiting." % (script_name, m, n, np.shape(W)[0], k, m, k), flush=True)
            sys.exit(1)

        if MEM == False:
            if XX == -1:
                print("%s: Using speed-optimized, not memory-optimized update to pNMF/opNMF. XX^T should be provided by the user. Exiting." % (script_name), flush=True)
                sys.exit(1)
            if np.shape(XX)[0] != np.shape(X)[0]:
                print("%s: (XX^T).shape = [%i, %i] | X.shape = [%i, %i]. XX should have shape [%i, %i]. Exiting." % (script_name, np.shape(XX)[0], np.shape(XX)[1], m, n, m, m), flush=True)
                sys.exit(1)
            if np.shape(XX)[1] != np.shape(X)[0]:
                print("%s: (XX^T).shape = [%i, %i] | X.shape = [%i, %i]. XX should have shape [%i, %i]. Exiting." % (script_name, np.shape(XX)[0], np.shape(XX)[1], m, n, m, m), flush=True)
                sys.exit(1)
        else:
            if X == -1:
                print("%s: Using memory-optimized, not speed-optimized update to pNMF/opNMF. X should be provided by the user. Exiting." % (script_name), flush=True)
                sys.exit(1)
                
    if OPNMF:
        if MEM == True:
            numerator = np.matmul(X, np.matmul(X.T, W))

            # denominator = np.matmul(W, np.matmul(np.matmul(W.T, X), np.matmul(X.T, W)))

            denominator = np.matmul(W, np.matmul(W.T, numerator))
        else:
            numerator = np.matmul(XX, W)
    
            # denominator = np.matmul(W, np.matmul(W.T, np.matmul(XX,W)))

            denominator = np.matmul(W, np.matmul(W.T, numerator))
    else:
        if MEM == True:
            numerator = 2 * np.matmul(X, np.matmul(X.T, W))

            denominator1 = np.matmul(W, np.matmul(np.matmul(W.T, X), np.matmul(X.T,W)))
            denominator2 = np.matmul(X, np.matmul(np.matmul(X.T,W), np.matmul(W.T,W)))
            denominator = np.add(denominator1, denominator2)
            
            del denominator1, denominator2
        else:
            numerator = 2 * np.matmul(XX, W)

            denominator1 = np.matmul(W, np.matmul(W.T, np.matmul(XX, W)))
            denominator2 = np.matmul(XX, np.matmul(W, np.matmul(W.T, W)))
            denominator = np.add(denominator1, denominator2)
            
            del denominator1, denominator2

    fraction = np.divide(numerator, denominator + EPSILON)
    if rho != 1.0:  #no need to waste computational power when not needed
        W = np.multiply(W, np.power(fraction,rho) )
    else:
        W = np.multiply(W, fraction)
    del fraction
    return W

def qr(X, rank = 100, script_name = script_name):
    """
    return QR decomposition of XX^T matrix provided as input
    """
    #sanity check: X must be 2D matrix
    X_shape = np.shape(X)
    if np.shape(X_shape)[0] != 2:
        print("%s: %s: qr: Expected %i-D matrix, but got %i-D matrix. Exiting." % (script_name, module_name, 2, np.shape(X_shape)[0]), flush=True)
        sys.exit(1)

    #sanity check: rank must be less than or equal to min(m,n)
    if rank > np.amin(X_shape):
        print("%s: %s: qr: rank (%i) must be less than or equal to smallest dimension of the input X matrix (%i). Exiting." % (script_name, module_name, rank, np.amin(X_shape)), flush=True)
        sys.exit(1)

    Q, R = scipy.linalg.qr(X, mode = 'economic')

    Q = Q[:, 0:rank]
    R = R[0:rank, :]

    return Q, R

def multiplicative_update_qr(W, Q, R, script_name = script_name, EPSILON = np.finfo(np.float32).eps, SANITY_CHECK_FLAG = True):
    """
    return the W from update rule using Q and R from multiplicative update instead of XX^T
    """
    if SANITY_CHECK_FLAG:
        #sanity check: dimension checks
        if np.shape(Q)[0] != np.shape(W)[0]:
            print("%s: %s: multiplicative_update_qr: wrong shape - Q has shape [%i, %i] and W has shape [%i, %i]. Exiting." % (script_name, module_name, 2, np.shape(Q)[0], np.shape(Q)[1], np.shape(W)[0], np.shape(W)[1]), flush=True)
            sys.exit(1)

        if np.shape(Q)[1] != np.shape(R)[0]:
            print("%s: %s: multiplicative_update_qr: wrong shape - Q has shape [%i, %i] and R has shape [%i, %i]. Exiting." % (script_name, module_name, 2, np.shape(Q)[0], np.shape(Q)[1], np.shape(R)[0], np.shape(R)[1]), flush=True)
            sys.exit(1)

        if np.shape(W)[0] != np.shape(R)[1]:
            print("%s: %s: multiplicative_update_qr: wrong shape - W has shape [%i, %i] and R has shape [%i, %i]. Exiting." % (script_name, module_name, 2, np.shape(W)[0], np.shape(W)[1], np.shape(R)[0], np.shape(R)[1]), flush=True)
            sys.exit(1)

    #if using X=QR
    #XX^TW in terms of QR \approx X
    XXTW = np.matmul(Q, np.matmul(R, np.matmul(np.transpose(R), np.matmul(np.transpose(Q), W))))
    W = np.multiply(W, np.divide( XXTW, np.matmul(W, np.matmul(np.transpose(W), XXTW) ) ) )
    del XXTW

    #if using XX^T=QR   #this will be infeasible with large X where m >> n because XX^T won't be storable in memory
    # W =  np.multiply( W, np.divide( np.matmul( Q, np.matmul(R, W) ), ( np.matmul( W, np.matmul( np.matmul( np.transpose(W), Q), np.matmul(R, W) ) ) ) + EPSILON ) )
    return W

def svd(X, rank = 100, script_name = script_name):
    """
    return SVD decomposition (U, s, Vh) of X matrix provided as input
    """
    #sanity check: X must be 2D matrix
    X_shape = np.shape(X)
    if np.shape(X_shape)[0] != 2:
        print("%s: %s: svd: Expected %i-D matrix, but got %i-D matrix. Exiting." % (script_name, module_name, 2, np.shape(X_shape)[0]), flush=True)
        sys.exit(1)

    #sanity check: rank must be less than or equal to m
    if rank > np.amin(X_shape):
        print("%s: %s: svd: rank (%i) must be less than or equal to smallest dimension of the input X matrix (%i). Exiting." % (script_name, module_name, rank, np.amin(X_shape)), flush=True)
        sys.exit(1)

    U, s, Vh = scipy.linalg.svd(X, full_matrices = False, lapack_driver = "gesvd")

    U = U[:, 0:rank]
    s = s[0:rank]
    Vh = Vh[0:rank, :]
    return U, s, Vh

def truncated_svd(X, rank = 100, n_iter = 5, n_oversamples = 10, random_state = 0, script_name = script_name):
    """
    return Truncated SVD (Latent Semantic Analysis LSA) decomposition (U, s, Vh) of X matrix provided as input
    """
    import sklearn.decomposition
    import importlib.metadata

    algorithm = "randomized"
    power_iteration_normalizer = "auto"

    #you need to have sklearn version 1.1 to use n_oversamples and power_iteration_normalizer
    #you need to have sklearn version 0.18.0 to use iterated_power, tol, randomized, and randomized
    scikit_learn_version = importlib.metadata.version('scikit-learn')
    scikit_learn_version_major = int(scikit_learn_version.split(".")[0])
    scikit_learn_version_minor = int(scikit_learn_version.split(".")[1])
    if (scikit_learn_version_major) < 1 and scikit_learn_version_minor < 18:
        print("%s: %s: randomized_pca: scikit-learn must be version 0.18.0 or higher to run randomized pca. Your current version of scikit-learn is %s" % (script_name, module_name, scikit_learn_version), flush=True)

    #sanity check: X must be 2D matrix
    X_shape = np.shape(X)
    if np.shape(X_shape)[0] != 2:
        print("%s: %s: truncated_svd: Expected %i-D matrix, but got %i-D matrix. Exiting." % (script_name, module_name, 2, np.shape(X_shape)[0]), flush=True)
        sys.exit(1)

    #sanity check: rank must be less than or equal to m
    if rank > np.amin(X_shape):
        print("%s: %s: truncated_svd: rank (%i) must be less than or equal to smallest dimension of the input X matrix (%i). Exiting." % (script_name, module_name, rank, np.amin(X_shape)), flush=True)
        sys.exit(1)


    if (scikit_learn_version_major) >= 1 and scikit_learn_version_minor >= 1:
        svd = sklearn.decomposition.TruncatedSVD(n_components=rank, algorithm = algorithm, n_iter = n_iter, n_oversamples = n_oversamples, power_iteration_normalizer = power_iteration_normalizer, random_state = random_state)
    else:
        svd = sklearn.decomposition.TruncatedSVD(n_components=rank, algorithm = algorithm, n_iter = n_iter, random_state = random_state)

    print("%s: %s: truncated_svd: initialized truncated svd object with %s algorithm, %i rank, %i iterations, %i oversampling, %s power iteration normalized, and %i random state. " % (script_name, module_name, algorithm, rank, n_iter, n_oversamples, power_iteration_normalizer, random_state), flush=True)

    print("%s: %s: truncated_svd: fitting to X" % (script_name, module_name), flush=True)
    U = svd.fit_transform(X)
    print("%s: %s: truncated_svd: fit to X" % (script_name, module_name), flush=True)

    s = svd.singular_values_
    Vh = svd.components_
    return U, s, Vh

def randomized_pca(X, rank = 100, iterated_power = 5, n_oversamples = 10, random_state = 0, script_name = script_name):
    """
    return randomized PCA decomposition (U, s, Vh) of X matrix provided as input
    """
    import sklearn.decomposition
    import importlib.metadata

    svd_solver = "randomized"
    power_iteration_normalizer = "auto"

    #you need to have sklearn version 1.1 to use n_oversamples and power_iteration_normalizer
    #you need to have sklearn version 0.18.0 to use iterated_power, tol, randomized, and randomized
    scikit_learn_version = importlib.metadata.version('scikit-learn')
    scikit_learn_version_major = int(scikit_learn_version.split(".")[0])
    scikit_learn_version_minor = int(scikit_learn_version.split(".")[1])
    if (scikit_learn_version_major) < 1 and scikit_learn_version_minor < 18:
        print("%s: %s: randomized_pca: scikit-learn must be version 0.18.0 or higher to run randomized pca. Your current version of scikit-learn is %s" % (script_name, module_name, scikit_learn_version), flush=True)

    #sanity check: X must be 2D matrix
    X_shape = np.shape(X)
    if np.shape(X_shape)[0] != 2:
        print("%s: %s: randomized_pca: Expected %i-D matrix, but got %i-D matrix. Exiting." % (script_name, module_name, 2, np.shape(X_shape)[0]), flush=True)
        sys.exit(1)

    #sanity check: rank must be less than or equal to m
    if rank > np.amin(X_shape):
        print("%s: %s: randomized_pca: rank (%i) must be less than or equal to smallest dimension of the input X matrix (%i). Exiting." % (script_name, module_name, rank, np.amin(X_shape)), flush=True)
        sys.exit(1)

    if (scikit_learn_version_major) >= 1 and scikit_learn_version_minor >= 1:
        pca = sklearn.decomposition.PCA(n_components=rank, svd_solver = svd_solver, iterated_power = iterated_power, n_oversamples = n_oversamples, power_iteration_normalizer = power_iteration_normalizer, random_state = random_state)
    else:
        pca = sklearn.decomposition.PCA(n_components=rank, svd_solver = svd_solver, iterated_power = iterated_power, random_state = random_state)        

    print("%s: %s: randomized_pca: initialized randomized pca object with %s algorithm, %i rank, %i iterations, %i oversampling, %s power iteration normalized, and %i random state. " % (script_name, module_name, svd_solver, rank, iterated_power, n_oversamples, power_iteration_normalizer, random_state), flush=True)

    print("%s: %s: randomized_pca: fitting to X" % (script_name, module_name), flush = True)
    U = pca.fit_transform(X)
    print("%s: %s: randomized_pca: fit to X" % (script_name, module_name), flush = True)

    Vh = pca.components_
    mean = pca.mean_
    return U, Vh, mean

def multiplicative_update_randpca(U, Vh, mean, script_name = script_name, EPSILON = np.finfo(np.float32).eps, SANITY_CHECK_FLAG = True):
    """
    return the W from update rule using U (reduced dimension of X), Vh (weights of pca), and mean (mean of X) from multiplicative update instead of XX^T
    """
    #To-Do: add sanity checks

    UV = (np.matmul(U, Vh) + mean)
    W =  np.multiply( W, np.divide( np.matmul( UV, np.matmul(np.transpose(UV), W) ), ( np.matmul( W, np.matmul( np.matmul( np.transpose(W), UV), np.matmul(np.transpose(UV), W) ) ) ) + EPSILON ) )
    del UV
    return W

def multiplicative_update_svd(W, US, script_name = script_name, EPSILON = np.finfo(np.float32).eps, SANITY_CHECK_FLAG = True):
    """
    return the W from update rule using Q and R from multiplicative update instead of XX^T
    expect US as np.matmul(U,S) of svd or truncated svd
    where S = s * np.eye(N = np.shape(s)[0]) with s being singular values
    """
    if SANITY_CHECK_FLAG:
        #sanity check: dimension checks
        if np.shape(W)[1] != np.shape(U)[0]:
            print("%s: %s: multiplicative_update_svd: wrong shape - W has shape [%i, %i] and U has shape [%i, %i]. Exiting." % (script_name, module_name, 2, np.shape(W)[0], np.shape(W)[1], np.shape(U)[0], np.shape(U)[1]), flush=True)
            sys.exit(1)

    W =  np.multiply( W, np.divide( np.matmul( US, np.matmul(np.transpose(US), W) ), ( np.matmul( W, np.matmul( np.matmul( np.transpose(W), US), np.matmul(np.transpose(US), W) ) ) ) + EPSILON ) )
    return W

def multiplicative_update_quadratic(X, W, MEM = True, OPNMF = False, rho = (1/3), XX = -1, EPSILON = np.finfo(np.float32).eps):
    """
    pnmf matlab mem: W = W .* (2X*(X'*W)) ./ (W*((W'*X)*(X'*W)) + X*(X'*W)*(W'*W));
    pnmf matlab original: W = W .* (2XX*W) ./ (W*(W'*XX*W) + XX*W*(W'*W));

    after experimenting, we noted that pNMF required the 2 in the numerator only if stabilization by normalization is not performed; if no normalization is performed and one performs update using convergence guaranteed update rule with cubic or fourth root, then leaving the 2 out will screw up the reconstruction error of pNMF

    I am keeing this function mainly to keep notes about the need for 2 in the numerator for pNMF. This should not be used as actual updates to NMF unless you have normalization step following the W update inside this function.

    Note that this is pNMF not opNMF by default; you need to turn OPNMF as true for it to be opNMF update with simpler denominator
    """
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    k = np.shape(W)[1]

    #sanity checks: dimension mismatches
    if np.shape(W)[0] != m:
        print("%s: X has shape [%i, %i], but W has shape [%i, %i]. W should have shape [%i, %i]. Exiting." % (script_name, m, n, np.shape(W)[0], k, m, k), flush=True)
        sys.exit(1)

    if MEM == False:
        if XX == -1:
            print("%s: Using speed-optimized, not memory-optimized update to pNMF/opNMF. XX^T should be provided by the user. Exiting." % (script_name), flush=True)
            sys.exit(1)
        if np.shape(XX)[0] != np.shape(X)[0]:
            print("%s: (XX^T).shape = [%i, %i] | X.shape = [%i, %i]. XX should have shape [%i, %i]. Exiting." % (script_name, np.shape(XX)[0], np.shape(XX)[1], m, n, m, m), flush=True)
            sys.exit(1)
        if np.shape(XX)[1] != np.shape(X)[0]:
            print("%s: (XX^T).shape = [%i, %i] | X.shape = [%i, %i]. XX should have shape [%i, %i]. Exiting." % (script_name, np.shape(XX)[0], np.shape(XX)[1], m, n, m, m), flush=True)
            sys.exit(1)

    if MEM == True:

        if OPNMF:
            numerator = np.matmul(X, np.matmul(X.T, W))
            denominator = np.matmul(W, np.matmul(W.T, numerator))
        else:
            numerator = 2 * np.matmul(X, np.matmul(X.T, W))
            denominator1 = np.matmul(W, np.matmul(np.matmul(W.T, X), np.matmul(X.T,W)))
            denominator2 = np.matmul(X, np.matmul(np.matmul(X.T,W), np.matmul(W.T,W)))
            denominator = np.add(denominator1, denominator2)
            del denominator1, denominator2
    else:
        if OPNMF:
            numerator = np.matmul(XX, W)
            denominator = np.matmul(W, np.matmul(W.T, numerator))

        else:
            numerator = 2 * np.matmul(XX, W)

            denominator1 = np.matmul(W, np.matmul(W.T, np.matmul(XX, W)))
            denominator2 = np.matmul(XX, np.matmul(W, np.matmul(W.T, W)))
            denominator = np.add(denominator1, denominator2)
            del denominator1, denominator2

    fraction = np.divide(numerator, denominator + EPSILON)

    W = np.multiply(W, np.power(fraction, rho) )
    del fraction
    return W

def reset_small_value(W, min_reset_value = 1.0e-16):
    """
    for elements in the provided matrix W that are less than min_reset_value, set them to min_reset_value
    """
    W[W<min_reset_value] = min_reset_value
    return W