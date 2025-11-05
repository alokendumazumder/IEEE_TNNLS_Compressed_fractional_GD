import os
import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
import torch
from tqdm import tqdm
from scipy.special import roots_jacobi
# Using latex for the axis of the plot
# plt.rcParams['text.usetex'] = True
import itertools

def prep():
    """
    Initializations
    """
    directory_list = ['result_exp_1', 'result_exp_2', 'result_exp_3', 'result_exp_4', 'figures']
    directory_dataset = 'dataset'
    if not os.path.exists(directory_dataset):
        os.makedirs(directory_dataset)
        raise Exception("No dataset detected")
    for dir in directory_list:
        if not os.path.exists(dir):
            os.makedirs(dir)

def cal_sum_rank_1(A):
    """
    Calculate the first part of the smoothness matrix (1/4 sum) and the threshold lambda
    Input:  dataset A
    Output: the first part of smoothness matrix, the minimum, maximum eigenvalue of it
    """
    n, d = A.shape
    sum = np.zeros((d, d))
    for i in range(n):
        A_i_col, A_i_row = A[i][:, np.newaxis], A[i][np.newaxis, :]
        sum += A_i_col @ A_i_row
    sum = (1 / (4 * n)) * sum
    lam_min = np.abs(np.min(sp.linalg.eigvals(sum)))
    lam_max = np.abs(np.max(sp.linalg.eigvals(sum)))
    # Return the sum of the rank one matrices with cofficient, and the minimum eigenvalue
    return [sum, lam_min, lam_max] #write braces

def sigmoid(x):
    """
    Sigmoid function used in the calculation
    Input:  a scalar x
    Output: the result of sigmoid function
    """
    return 1 / (1 + np.exp(-x))

def cal_matrix_norm(D, x):
    """
    Calculate matrix norm for the current iterate
    Input:  a PSD matrix D, and a vector x of corresponging length
    Output: square matrix norm of the vector
    """
    return (x[np.newaxis, :] @ D @ x[:, np.newaxis])[0][0]

def plot_eigs(L):
    """
    Plot the eigenvalue distribution of the smoothness matrix L
    Input: the smoothness matrix L
    """
    eig_vals_L, eig_vecs_L = np.linalg.eigh(L)
    num_bins = int(len(eig_vals_L)) * 2
    plt.hist(eig_vals_L, bins=num_bins)
    plt.xlabel("Eigenvalues of smoothness matrix")
    plt.ylabel("Number of eigenvalues")
    plt.show()

def cal_grad(A, b, x, lam):
    """
    Calculate the gradient of the for logistic regression with non-convex regularizer
    Input:  A is the dataset, b is the corresponding labels, x is the iterate, lam is the value of lambda
    Output: the gradient of function with non-convex regularizer in this case
    """
    n, d = A.shape
    # Calculating grad of g(x):
    inner_mat = A * x
    inner_slr = np.sum(inner_mat, axis=1)
    inner_slr_b = -b * inner_slr
    inner_sig = sigmoid(inner_slr_b)
    inner_sig_b = -b * inner_sig
    inner_grad_mat = inner_mat * inner_sig_b[:, np.newaxis]
    grad_g = np.average(inner_grad_mat, axis=0)
    # Calculating grad of r(x):
    numer = 2 * x
    denom = (1 + x ** 2) ** 2
    grad_r = numer / denom
    return grad_g + lam * grad_r

def hessian_full(x, A, b, lamb):
    n_samples, n_features = A.shape
    
    # Logistic loss component
    z = A @ x  # z = A * x (shape: (n_samples,))
    
    # Probabilities p = sigmoid(b * z)
    p = 1 / (1 + np.exp(b * z))  # (shape: (n_samples,))
    
    # Weights w_i = p_i * (1 - p_i)
    W = p * (1 - p)  # (shape: (n_samples,))
    
    # Hessian of logistic loss: A^T * diag(W) * A
    H_logistic = (A.T * W) @ A / n_samples  # Efficient computation of A^T W A
    
    # Non-convex regularizer Hessian (diagonal)
    diag_hessian = 2 * lamb * (1 - x**2) / (1 + x**2)**3
    
    # The regularizer Hessian is diagonal, so we use np.diag for a diagonal matrix
    H_regularizer = np.diag(diag_hessian)
    
    # Full Hessian: sum of the logistic loss Hessian and the regularizer Hessian
    H_full = H_logistic + H_regularizer
    
    return H_full

def gauss_jacobi_quadrature(s,a,b):
    nodes, weights = roots_jacobi(s, a, b)
    return nodes, weights

def frac_grad(x, A, b, lamb, delta_f, nodes, weights, alpha, s, c, beta):
    
    C = (1-alpha)*(2**(alpha-1))

    n = len(delta_f)
    d = np.zeros_like(delta_f)
    if beta !=0:
        H_f = hessian_full(x,A,b,lamb)
    else:
        pass

    for j in range(n):

        grad_sum = 0
        hess_sum = 0

        for l in range(s):

            y_l = (abs(x[j] - c[j])/2)*(1+nodes[l]) + c[j]
            x_perturbed = x.copy()
            x_perturbed[j] +=(y_l - x[j])

            grad_sum += weights[l]*delta_f[j]
            if beta !=0:
                hess_sum += weights[l]*H_f[j,j]
            else:
                pass

        d[j] = C*grad_sum + C*beta*abs(x[j] - c[j])*hess_sum

    return d

def rand_t_sparsifier(x, t):
    """
    Random t sparsifier
    Input:  vector x, the minibatch size t
    Output: vector x after rand-t sparsifier
    """
    d = x.shape[0]
    if t > d:
        raise AssertionError("Exceeding minibatch")
    mask = np.zeros_like(x)
    idx = np.random.choice(np.arange(d), size=t, replace=False)
    mask[idx] = d / t
    return mask * x

def special_rand_1(x, idx):
    """
    Specially designed rand-1 sparisifer to guarantee same randomness
    Input:  x is the vector, idx is a list of indices specifyinng the coordinates picked
    Output: x after applying rand-1 sparidifer
    """
    mask = np.zeros_like(x)
    mask[idx] = d
    return mask * x

def cal_func_val(A, b, x, lam):
    """
    Calculate the function value
    Input:  A is the dataset, b is the corresponding label, x is the vector where we are evaluating
            the function value, lam is the value of lambda we used for the non-convex regularizer
    Output: the function value at vector x
    """
    # Function value of g part
    temp_1 = np.sum(A * x, axis=1)
    temp_2 = temp_1 * (-b)
    temp_3 = np.log(np.exp(temp_2) + 1)
    g_value = np.average(temp_3)
    # Function value of r part
    numerator = x ** 2
    denominator = 1 + x ** 2
    r_value = np.sum(numerator / denominator)
    return g_value + lam * r_value

# Functions for experiment 1, results can be found in folder './result_exp_1'
def fgd(alpha, s,beta, tune, nodes, weights, x_iter, L_scalar, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running FGD with scalar stepsize, scalar smoothness constant, rand-1 sketch is used
    The average norm is solved as "logistic_exp_1_curve_1_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # x_iter_list = []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_0     = (1 / (d * L_scalar))
    D           = np.eye(d) * gamma_0
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        # hess_x = hessian_full(x_iter, A, b, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - gamma_0 * frac_gradient
        # Recording
        # print(np.linalg.norm(grad_x),np.linalg.norm(frac_gradient))
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    # Saving
    result_file = "logistic_exp_1_curve_1_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def fgd_mat(alpha, s,beta, tune, nodes, weights, x_iter, L_scalar, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CFGD with scalar stepsize, matrix smoothness constant, rand-1 sketch is used
    The average norm is solved as "logistic_exp_1_curve_1_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # x_iter_list = []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_1     = (1 / (d * L_scalar))
    D           = np.eye(d) * gamma_1
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        # hess_x = hessian_full(x_iter, A, b, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - gamma_1 * frac_gradient
        # Recording
        # print(np.linalg.norm(grad_x),np.linalg.norm(frac_gradient))
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    # Saving
    result_file = "logistic_exp_1_curve_2_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def cfgd(alpha, s,beta, tune, nodes, weights, x_iter, L_scalar, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CFGD with scalar stepsize, scalar smoothness constant, rand-1 sketch is used
    The average norm is solved as "logistic_exp_1_curve_1_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # x_iter_list = []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_0     = (1 / (d * L_scalar))
    D           = np.eye(d) * gamma_0
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        # hess_x = hessian_full(x_iter, A, b, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - gamma_0 * special_rand_1(frac_gradient, idx=indices[i])
        # Recording
        # print(np.linalg.norm(grad_x),np.linalg.norm(frac_gradient))
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    # Saving
    result_file = "logistic_exp_1_curve_3_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def cfgd_1(alpha, s,beta, tune, nodes, weights, x_iter, L_scalar, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CFGD with scalar stepsize, matrix smoothness constant, rand-1 sketch is used
    The average norm is solved as "logistic_exp_1_curve_1_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # x_iter_list = []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_1     = (1 / (d * L_scalar))
    D           = np.eye(d) * gamma_1
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        # hess_x = hessian_full(x_iter, A, b, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - gamma_1 * special_rand_1(frac_gradient, idx=indices[i])
        # print(np.count_nonzero(special_rand_1(frac_gradient, idx=indices[i]).shape))
        # Recording
        # print(np.linalg.norm(grad_x),np.linalg.norm(frac_gradient))
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    # Saving
    result_file = "logistic_exp_1_curve_4_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def cfgd_alg_1_L_diag_inv(alpha, s,beta, tune, nodes, weights, x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with diagonal matrix stepsize, matrix smoothness, rand-1, for algorithm 1
    The average norm is solved as "logistic_exp_1_curve_3_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # x_iter_list = []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_2     = 1 / d
    D           = L_matrix * gamma_2
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_x = cal_grad(A, b, x_iter, lam)
        # hess_x = hessian_full(x_iter, A, b, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - D @ special_rand_1(frac_gradient, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 3", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_5_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def cfgd_alg_1_L_inv(alpha, s,beta, tune, nodes, weights, x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with L-1 matrix stepsize, matrix smoothness, rand-1, for algorithm 1
    The average norm is solved as "logistic_exp_1_curve_7_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # x_iter_list = []
    # Get shape
    n, d = A.shape
    # Get L related matrix
    L_half      = sp.linalg.fractional_matrix_power(L_matrix, 1/2)
    L_inv       = sp.linalg.inv(L_matrix)
    L_inv_diag  = np.diag(np.diag(L_inv))
    # Get stepsize matrix
    gamma_3     = (1 / (np.linalg.eigh(L_half @ L_inv_diag @ L_half)[0][-1])) * (1/d)
    D           = L_inv * gamma_3
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_x = cal_grad(A, b, x_iter, lam)
        # hess_x = hessian_full(x_iter, A, b, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - D @ special_rand_1(frac_gradient, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))

    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 7", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_6_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def cfgd_alg_1_L_half_inv(alpha, s,beta, tune, nodes, weights, x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with L-1/2 matrix stepsize, matrix smoothness, rand-1, for algorithm 1
    The average norm is solved as "logistic_exp_1_curve_7_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # x_iter_list = []
    # Get shape
    n, d = A.shape
    # Get L related matrix
    L_half      = sp.linalg.fractional_matrix_power(L_matrix, 1/2)
    L_half_inv  = sp.linalg.fractional_matrix_power(L_matrix, -(1/2))
    # Get stepsize matrix
    gamma_4     = (1 / (np.linalg.eigh(L_half)[0][-1])) * (1/d)
    D           = L_half_inv * gamma_4
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_x = cal_grad(A, b, x_iter, lam)
        # hess_x = hessian_full(x_iter, A, b, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - D @ special_rand_1(frac_gradient, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))

    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 7", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_7_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def cfgd_alg_2_L_diag_inv(alpha, s,beta, tune, nodes, weights, x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with diagonal matrix stepsize, matrix smoothness, rand-1, for algorithm 2
    The average norm is solved as "logistic_exp_1_curve_3_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # x_iter_list = []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_5     = 1 / d
    D           = L_matrix * gamma_5
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_x = cal_grad(A, b, x_iter, lam)
        # hess_x = hessian_full(x_iter, A, b, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - special_rand_1(D @ frac_gradient, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 3", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_8_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def cfgd_alg_2_L_diag_inv_ablation(alpha, s,beta, tune, nodes, weights, x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with diagonal matrix stepsize, matrix smoothness, rand-1, for algorithm 2
    The average norm is solved as "logistic_exp_1_curve_3_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # x_iter_list = []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_5     = 1 / d
    D           = L_matrix * gamma_5
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_x = cal_grad(A, b, x_iter, lam)
        # hess_x = hessian_full(x_iter, A, b, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - special_rand_1(D @ frac_gradient, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 3", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "cfgd_diag" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}_alpha_{}.npy".format(lam, seed,alpha)
    outfile = os.path.join("result_exp_6", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cfgd_curve_1_exp_2(alpha, s,beta, tune, nodes, weights, x_iter, L_scalar, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null', t=0):
    """
    Running CGD with scalar stepsize, scalar smoothness constant, rand-t
    The average norm is solved as "logistic_exp_2_curve_1_[DATASET]_lam_[LAMBDA]_rand_[T]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_1         = (t / d) * (1 / L_scalar)
    D               = np.eye(d) * gamma_1
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - gamma_1 * rand_t_sparsifier(frac_gradient, t)
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 2 curve 1", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_2_curve_1_" + dataset.split(sep='.')[0] + \
                  "_lam_{}_rand_{}_seed_{}.npy".format(lam, t, seed)
    outfile = os.path.join("result_exp_2", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cfgd_curve_2_exp_2(alpha, s,beta, tune, nodes, weights, x_iter, L, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null', t=0):
    """
    Running CGD with scalar stepsize, matrix smoothness, rand-t
    The average norm is solved as "logistic_exp_2_curve_2_[DATASET]_lam_[LAMBDA]_rand_[T]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix
    L_diag = np.diag(np.diag(L))
    L_comb = ((d - t)/(d - 1)) * L_diag + ((t - 1)/(d - 1)) * L
    max_eig_L_comb = np.linalg.eigh(L_comb)[0][-1]
    # Get stepsize matrix
    gamma_2         = (t / d) * (1 / max_eig_L_comb)
    D = np.eye(d) * gamma_2
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - gamma_2 * rand_t_sparsifier(frac_gradient, t)
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 2 curve 2", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_2_curve_2_" + dataset.split(sep='.')[0] + \
                  "_lam_{}_rand_{}_seed_{}.npy".format(lam, t, seed)
    outfile = os.path.join("result_exp_2", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cfgd_curve_3_exp_2(alpha, s,beta, tune, nodes, weights, x_iter, L, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null', t=0):
    """
    Running algorithm 1 with matrix stepsize, matrix smoothness, rand-t
    The average norm is solved as "logistic_exp_2_curve_4_[DATASET]_lam_[LAMBDA]_rand_[T]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix
    L_diag = np.diag(np.diag(L))
    L_comb = (d / t) * (((d - t) / (d - 1)) * L_diag + ((t - 1) / (d - 1)) * L)
    # Get stepsize matrix
    D =  np.linalg.inv(L_comb)
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - D @ rand_t_sparsifier(frac_gradient, t)
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 2 curve 4", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_2_curve_3_" + dataset.split(sep='.')[0] + \
                  "_lam_{}_rand_{}_seed_{}.npy".format(lam, t, seed)
    outfile = os.path.join("result_exp_2", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cfgd_curve_4_exp_2(alpha, s,beta, tune, nodes, weights, x_iter, L, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null', t=0):
    """
    Running algorithm 2 with optimal matrix stepsize, matrix smoothness, rand-t
    The average norm is solved as "logistic_exp_2_curve_3_[DATASET]_lam_[LAMBDA]_rand_[T]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix
    L_diag = np.diag(np.diag(L))
    L_comb = (d / t) * (((d - t) / (d - 1)) * L_diag + ((t - 1) / (d - 1)) * L)
    # Get stepsize matrix
    D =  np.linalg.inv(L_comb)
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        c = x_iter + tune*grad_x
        frac_gradient = frac_grad(x_iter,A, b, lam, grad_x, nodes, weights, alpha, s, c, beta)
        x_iter = x_iter - rand_t_sparsifier(D @ frac_gradient, t)
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 2 curve 3", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_2_curve_4_" + dataset.split(sep='.')[0] + \
                  "_lam_{}_rand_{}_seed_{}.npy".format(lam, t, seed)
    outfile = os.path.join("result_exp_2", result_file)
    np.save(outfile, arrnm)
    return arrnm


# Functions for experiment 3, results can be found in folder './result_exp_3'
def run_cgd_curve_1_exp_3(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with scalar stepsize, matrix smoothness, rand-1-uniform
    The average norm is solved as "logistic_exp_3_curve_1_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix
    L_diag = np.diag(np.diag(L_matrix))
    max_eig_L_diag = np.linalg.eigh(L_diag)[0][-1]
    # Get stepsize matrix
    gamma_1 = (1 / d) * (1 / max_eig_L_diag)
    D = np.eye(d) * gamma_1
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - gamma_1 * special_rand_1(grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 3 curve 1", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_3_curve_1_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_3", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_2_exp_3(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with scalar stepsize, matrix smoothness, rand-1-importance
    The average norm is solved as "logistic_exp_3_curve_1_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """

    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix and probability
    L_tr    = np.trace(L_matrix)
    prob_p  = np.diag(L_matrix) / L_tr
    indices_imp_curve_2 = np.random.choice(d, size=iterations, replace=True, p=prob_p)
    # Get stepsize matrix
    gamma_2 = 1 / L_tr
    D = np.eye(d) * gamma_2
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - gamma_2 * special_rand_1(grad_x, idx=indices_imp_curve_2[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 3 curve 2", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_3_curve_2_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_3", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_3_exp_3(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running algorithm 1 with diagonal matrix stepsize, matrix smoothness, rand-1-uniform/importance
    The average norm is solved as "logistic_exp_2_curve_3_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix and probability
    L_diag = np.diag(np.diag(L_matrix))
    L_diag_inv = np.linalg.inv(L_diag)
    # Get stepsize matrix
    D = (1 / d) * L_diag_inv
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - D @ special_rand_1(grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 3 curve 3", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_3_curve_3_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_3", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_5_exp_3(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running algorithm 2 with diagonal matrix stepsize, matrix smoothness, rand-1-uniform/importance
    The average norm is solved as "logistic_exp_2_curve_3_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix and probability
    L_diag = np.diag(np.diag(L_matrix))
    L_diag_inv = np.linalg.inv(L_diag)
    # Get stepsize matrix
    D = (1 / d) * L_diag_inv
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - special_rand_1(D @ grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 3 curve 4", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_3_curve_5_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_3", result_file)
    np.save(outfile, arrnm)
    return arrnm


# Functions for experiment 4, results can be found in folder './result_exp_4'
def find_min_GD(A, b, L, lam=0, iterations=200, plot=False, whole=False, index=-1):
    """
    Running GD to find the minimum of functions if we can not deduce its minimum
    Input:  iterations is the number of epochs of GD
    Output: the minimum encountered in the entire run of GD
    """
    # Shape
    n, d = A.shape
    # Iterate
    x_iter = np.ones(d)
    # Record
    fvals = []
    # Stepsize
    gamma = 1 / L
    min_f = 1000000
    for i in range(iterations):
        grad_x = cal_grad(A, b, x_iter, lam=lam)
        x_iter = x_iter - gamma * grad_x
        fvals.append(cal_func_val(A, b, x_iter, lam=lam))
        f_val = cal_func_val(A, b, x_iter, lam=lam)
        if min_f > f_val:
            min_f = f_val

    # Plotting to see if it is correct
    arrfv = np.array(fvals)
    if plot == True:
        iter_range = np.arange(1, arrfv.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Log Function values")
        plt.plot(iter_range, np.log10(arrfv), label="GD to find minimum", marker='o', markevery=500)
        # plt.legend()
        plt.show()
    return min_f

def DCGD(x_iter, gamma, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    DCGD scalar stepsize, scalar smoothness
    The average norm is solved as "logistic_exp_4_curve_1_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get stepsize matrix
    D = gamma * np.eye(d)
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            # Compressed
            comp_grad_c = rand_t_sparsifier(grad_x_c, t=1)
            grad_est += comp_grad_c
        # Get average
        grad_est = grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - gamma * grad_est
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    result_file = "logistic_exp_4_curve_1_" + dataset.split(sep='.')[0]\
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    outfile = os.path.join("result_exp_4", result_file)
    np.save(outfile, arrnm)
    return arrnm

def DCGD_mat(x_iter, gamma, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    DCGD scalar stepsize, matrix smoothness
    The average norm is solved as "logistic_exp_4_curve_1_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get stepsize matrix
    D = gamma * np.eye(d)
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            # Compressed
            comp_grad_c = rand_t_sparsifier(grad_x_c, t=1)
            grad_est += comp_grad_c
        # Get average
        grad_est = grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - gamma * grad_est
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    result_file = "logistic_exp_4_curve_2_" + dataset.split(sep='.')[0]\
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    outfile = os.path.join("result_exp_4", result_file)
    np.save(outfile, arrnm)
    return arrnm

def DCFGD(alpha, s, beta, tune, nodes, weights, x_iter, gamma, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    DCGD scalar stepsize, scalar smoothness
    The average norm is solved as "logistic_exp_4_curve_1_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get stepsize matrix
    D = gamma * np.eye(d)
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        frac_grad_est = np.zeros_like(x_iter)
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Frac gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            c_t = x_iter + tune*grad_x_c
            frac_grad_x_c = frac_grad(x_iter, split_A[c], split_b[c], lam, grad_x_c, nodes, weights, alpha, s, c_t, beta)
            # Gradient for each client
            # grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            # Compressed
            comp_frac_grad_c = rand_t_sparsifier(frac_grad_x_c, t=1)
            # comp_grad_c = rand_t_sparsifier(grad_x_c, t=1)
            # grad_est += comp_grad_c
            frac_grad_est += comp_frac_grad_c
        # Get average
        frac_grad_est = frac_grad_est / num_client
        # grad_est = grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - gamma * frac_grad_est
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    result_file = "logistic_exp_4_curve_3_" + dataset.split(sep='.')[0]\
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    outfile = os.path.join("result_exp_4", result_file)
    np.save(outfile, arrnm)
    return arrnm

def DCGD_1(x_iter, D, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    Running algorithm 1 with optimal diagonal matrix stepsize, matrix smoothness
    Matrix norm result:
    The average norm is solved as "logistic_exp_4_curve_3_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    Standard Euclidean norm result:
    The average norm is solved as "std_logistic_exp_4_curve_3_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm, stdnm, avgstd = [], [], [], []
    # Get stepsize matrix
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            # Compressed
            comp_grad_c = rand_t_sparsifier(grad_x_c, t=1)
            grad_est += comp_grad_c
        # Get average
        grad_est = grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - D @ grad_est
        # Recording matrix norm
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
        # Recording euclidean norm
        stdm_iter = np.linalg.norm(grad_x) ** 2
        stdnm.append(stdm_iter)
        avgstd.append(np.average(stdnm))
    # Plotting
    arrnm = np.array(avgnm)
    arrstd = np.array(avgstd)
    # if plot == True:
    #     iter_range = np.arange(1, arrnm.shape[0] + 1)
    #     plt.xlabel("Iterations")
    #     plt.ylabel("Lognorm")
    #     plt.yscale("log")
    #     plt.plot(iter_range, arrnm, label="s1", marker='o', markevery=2000)
    #     plt.plot(iter_range, arrstd, label="s2", marker='o', markevery=2000)
    #     plt.legend()
    #     plt.show()
    # Saving
    result_file_1 = "logistic_exp_4_curve_4_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    result_file_2 = "std_logistic_exp_4_curve_4_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    outfile_1 = os.path.join("result_exp_4", result_file_1)
    outfile_2 = os.path.join("result_exp_4", result_file_2)
    np.save(outfile_1, arrnm)
    np.save(outfile_2, arrstd)
    return arrnm, arrstd

def DCFGD_1(alpha, s, beta, tune, nodes, weights, x_iter, D, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    Running algorithm 1 with optimal diagonal matrix stepsize, matrix smoothness
    Matrix norm result:
    The average norm is solved as "logistic_exp_4_curve_3_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    Standard Euclidean norm result:
    The average norm is solved as "std_logistic_exp_4_curve_3_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm, stdnm, avgstd = [], [], [], []
    # Get stepsize matrix
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        frac_grad_est = np.zeros_like(x_iter)
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            c_t = x_iter + tune*grad_x_c
            frac_grad_x_c = frac_grad(x_iter, split_A[c], split_b[c], lam, grad_x_c, nodes, weights, alpha, s, c_t, beta)
            # Compressed
            comp_frac_grad_c = rand_t_sparsifier(frac_grad_x_c, t=1)
            # comp_grad_c = rand_t_sparsifier(grad_x_c, t=1)
            # grad_est += comp_grad_c
            frac_grad_est += comp_frac_grad_c
        # Get average
        frac_grad_est = frac_grad_est / num_client
        # grad_est = grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - D @ frac_grad_est
        # Recording matrix norm
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
        # Recording euclidean norm
        stdm_iter = np.linalg.norm(grad_x) ** 2
        stdnm.append(stdm_iter)
        avgstd.append(np.average(stdnm))
    # Plotting
    arrnm = np.array(avgnm)
    arrstd = np.array(avgstd)
    result_file_1 = "logistic_exp_4_curve_5_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    result_file_2 = "std_logistic_exp_4_curve_5_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    outfile_1 = os.path.join("result_exp_4", result_file_1)
    outfile_2 = os.path.join("result_exp_4", result_file_2)
    np.save(outfile_1, arrnm)
    np.save(outfile_2, arrstd)
    return arrnm, arrstd

def DCGD_2(x_iter, D, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    Running algorithm 2 with optimal diagonal matrix stepsize, matrix smoothness
    Matrix norm result:
    The average norm is solved as "logistic_exp_4_curve_4_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    Standard Euclidean norm result:
    The average norm is solved as "std_logistic_exp_4_curve_4_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm, stdnm, avgstd = [], [], [], []
    # Get stepsize matrix
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            # Compressed
            comp_grad_c = rand_t_sparsifier(D @ grad_x_c, t=1)
            grad_est += comp_grad_c
        # Get average
        grad_est = grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - grad_est
        # Recording matrix norm
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
        # Recording euclidean norm
        stdm_iter = np.linalg.norm(grad_x) ** 2
        stdnm.append(stdm_iter)
        avgstd.append(np.average(stdnm))
    # Plotting
    arrnm = np.array(avgnm)
    arrstd = np.array(avgstd)

    result_file_1 = "logistic_exp_4_curve_6_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    result_file_2 = "std_logistic_exp_4_curve_6_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    outfile_1 = os.path.join("result_exp_4", result_file_1)
    outfile_2 = os.path.join("result_exp_4", result_file_2)
    np.save(outfile_1, arrnm)
    np.save(outfile_2, arrstd)
    return arrnm, arrstd

def DCFGD_2(alpha, s, beta, tune, nodes, weights, x_iter, D, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    Running algorithm 2 with optimal diagonal matrix stepsize, matrix smoothness
    Matrix norm result:
    The average norm is solved as "logistic_exp_4_curve_4_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    Standard Euclidean norm result:
    The average norm is solved as "std_logistic_exp_4_curve_4_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm, stdnm, avgstd = [], [], [], []
    # Get stepsize matrix
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        frac_grad_est = np.zeros_like(x_iter)
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            c_t = x_iter + tune*grad_x_c
            frac_grad_x_c = frac_grad(x_iter, split_A[c], split_b[c], lam, grad_x_c, nodes, weights, alpha, s, c_t, beta)
            # Compressed
            comp_frac_grad_c = rand_t_sparsifier(D @ frac_grad_x_c, t=1)
            # comp_grad_c = rand_t_sparsifier(D @ grad_x_c, t=1)
            # grad_est += comp_grad_c
            frac_grad_est += comp_frac_grad_c
        # Get average
        frac_grad_est = frac_grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - frac_grad_est
        # Recording matrix norm
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
        # Recording euclidean norm
        stdm_iter = np.linalg.norm(grad_x) ** 2
        stdnm.append(stdm_iter)
        avgstd.append(np.average(stdnm))
    # Plotting
    arrnm = np.array(avgnm)
    arrstd = np.array(avgstd)
    return arrnm, arrstd

def DCFGD_2_ablation(alpha, s, beta, tune, nodes, weights, x_iter, D, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    Running algorithm 2 with optimal diagonal matrix stepsize, matrix smoothness
    Matrix norm result:
    The average norm is solved as "logistic_exp_4_curve_4_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    Standard Euclidean norm result:
    The average norm is solved as "std_logistic_exp_4_curve_4_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm, stdnm, avgstd = [], [], [], []
    # Get stepsize matrix
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        frac_grad_est = np.zeros_like(x_iter)
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            c_t = x_iter + tune*grad_x_c
            frac_grad_x_c = frac_grad(x_iter, split_A[c], split_b[c], lam, grad_x_c, nodes, weights, alpha, s, c_t, beta)
            # Compressed
            comp_frac_grad_c = rand_t_sparsifier(D @ frac_grad_x_c, t=1)
            # comp_grad_c = rand_t_sparsifier(D @ grad_x_c, t=1)
            # grad_est += comp_grad_c
            frac_grad_est += comp_frac_grad_c
        # Get average
        frac_grad_est = frac_grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - frac_grad_est
        # Recording matrix norm
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
        # Recording euclidean norm
        stdm_iter = np.linalg.norm(grad_x) ** 2
        stdnm.append(stdm_iter)
        avgstd.append(np.average(stdnm))
    # Plotting
    arrnm = np.array(avgnm)
    arrstd = np.array(avgstd)
    result_file_1 = "ablation_dcfgd2_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}_alpha_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq, alpha)
    result_file_2 = "std_logistic_exp_4_curve_7_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    outfile_1 = os.path.join("result_exp_5", result_file_1)
    outfile_2 = os.path.join("result_exp_5", result_file_2)
    np.save(outfile_1, arrnm)
    np.save(outfile_2, arrstd)
    return arrnm, arrstd

import numpy as np
import scipy as sp

def det_CGD2_VR(x_iter, L_mat, num_client, iterations=100, lam=0,p=0.2,t=1):
    # Recording norms per iteration
    norms, avgnm, stdnm, avgstd = [], [], [], []

    # fractional‐power matrix and its det for normalization
    d = x_iter.shape[0]
    L_inv = sp.linalg.inv(L_mat)
    L_diag = np.diag(np.diag(L_mat))
    L_diag_inv = sp.linalg.inv(L_diag)
    L_inv_half = sp.linalg.fractional_matrix_power(L_mat, -0.5)
    L_inv_exp = (d/t)*L_diag_inv
    lam_exp = np.linalg.eigh(L_inv_exp - L_diag_inv)[0][-1] # The lambda_max of expectation
    alpha = (1 - p) / (num_client * p)
    lambda_max_list_Li = np.array([np.abs(np.max(sp.linalg.eigvals(L_list[i]))) for i in range(num_client)])
    lambda_max_list_LLi = np.array(
        [np.abs(np.max(sp.linalg.eigvals(L_inv_half @ L_list[i] @ L_inv_half))) for i in range(num_client)]
    )
    beta_lam = np.average(lambda_max_list_Li * lambda_max_list_LLi)
    coeff = 2 / (1 + np.sqrt(1 + 4 * alpha * beta_lam * lam_exp))
    D = coeff * L_diag_inv
    # To avoide the overflow caused by determinant
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    # D_1d     = sp.linalg.fractional_matrix_power(D, 1.0 / d)
    D_1d_det = sp.linalg.det(D_1d)

    # initialize global estimator g⁰ = D @ ∇f(x⁰)
    grad_x = cal_grad(A, b, x_iter, lam)
    g_k     = D @ grad_x

    for _ in range(iterations):
        # sample cₖ ∼ Bernoulli(p)
        c_k = np.random.binomial(1, p)

        # each worker computes its new estimator
        g_next_list = []
        for i in range(num_client):
            # local update step
            x_next = x_iter - g_k

            # new/old local gradients
            grad_i_new = cal_grad(split_A[i], split_b[i], x_next, lam)
            grad_i_old = cal_grad(split_A[i], split_b[i], x_iter,  lam)

            if c_k == 1:
                # full refresh
                g_i_next = D @ grad_i_new
            else:
                # variance‐reduced update
                delta    = grad_i_new - grad_i_old
                g_i_next = g_k + rand_t_sparsifier(D @ delta,t=1)

            g_next_list.append(g_i_next)

        # average to form new global estimator g^{k+1}
        g_k    = sum(g_next_list) / num_client
        x_iter = x_next

        # compute and record norms
        grad_x    = cal_grad(A, b, x_iter, lam)
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.mean(norms))

        std_iter = np.linalg.norm(grad_x)**2
        stdnm.append(std_iter)
        avgstd.append(np.mean(stdnm))

    # assemble results
    arrnm  = np.array(avgnm)
    arrstd = np.array(avgstd)
    result_file_1 = "logistic_exp_cgd_vr_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}_prob_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq, p)
    # result_file_2 = "std_logistic_exp_4_curve_5_" + dataset.split(sep='.')[0] \
    #               + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq, p)
    outfile_1 = os.path.join("result_exp_8", result_file_1)
    # outfile_2 = os.path.join("result_exp_4", result_file_2)
    np.save(outfile_1, arrnm)
    # np.save(outfile_2, arrstd)
    return arrnm


# Parameters
parser = argparse.ArgumentParser(description='Experiment')
parser.add_argument('--exp_index', '-i', help="Index of the experiment", default=1, required=True)
parser.add_argument('--dataset', '-d', help='Name of dataset', default='a1a', required=True)
parser.add_argument('--lambda_val', '-l', help='Value of lambda', default=0.1)
parser.add_argument('--alpha_val', '-a', help='Value of alpha (should be b/w 0 to 1)', default=0.2)
parser.add_argument('--beta_val', '-b', help='Value of beta (should be a real number)', default=0)
parser.add_argument('--tune_val', '-t', help='Value of coeff in guided gradient process', default=-0.0675) #-0.0675
parser.add_argument('--GJQ', '-p', help='Number of points needed from GJQ', default=1)
parser.add_argument('--GJQ_coeff_a', '-pa', help='Left coeff of GJQ', default= 0.8) #-0.2, -0.1 (best) for ditributed i guess !
parser.add_argument('--GJQ_coeff_b', '-pb', help='Right coeff of GJQ', default= 0.9)
parser.add_argument('--client', '-c', help='Number of client', default=1)
parser.add_argument('--seed', '-s', default=522)
parser.add_argument('--eps', '-e', help='Error level', default=0.0001)
parser.add_argument('--iterations', '-K', help='Number of iterations to run', default=10000)
args = parser.parse_args()


if __name__ == "__main__":
    prep()

    # Hyperparameters
    cur_exp = int(args.exp_index)
    iterations = int(args.iterations)
    lam = float(args.lambda_val)
    alpha = float(args.alpha_val)
    beta = float(args.beta_val)
    tune = float(args.tune_val)
    s = int(args.GJQ)
    a = float(args.GJQ_coeff_a)
    b_ = float(args.GJQ_coeff_b)
    num_client = int(args.client)
    epsilon_sq = float(args.eps)
    dataset = args.dataset + '.txt'
    PLOT = False


    # Load the a1a dataset, for a1a, A: (1605, 119); b: (1605, )
    path_dataset = os.path.join('dataset', dataset)
    A_train_a1a, b_train_a1a = load_svmlight_file(path_dataset)

    # Turning into numpy arrays
    A, b = A_train_a1a.toarray(), b_train_a1a
    n, d = A.shape
    nodes, weights = gauss_jacobi_quadrature(s,a,b_)
    # Calculating L smoothness matrix
    sum_mat, lam_min, lam_max = cal_sum_rank_1(A)[0], cal_sum_rank_1(A)[1], cal_sum_rank_1(A)[2]

    # Constructing L using different lambda
    L = sum_mat + 2 * lam * np.eye(d)

    # Show eigen value histogram
    if PLOT == True:
        # plot_eigs(L)
        pass

    # Calculating related matrices and scalars
    L_diag          = np.diag(np.diag(L))
    L_diag_inv      = sp.linalg.inv(L_diag)
    L_inv           = sp.linalg.inv(L)
    L_inv_det       = sp.linalg.det(L_inv)
    L_half          = sp.linalg.fractional_matrix_power(L, (1 / 2))
    L_half_inv      = sp.linalg.inv(L_half)

    max_eig_L       = np.linalg.eigh(L)[0][-1]
    max_eig_L_diag  = np.linalg.eigh(L_diag)[0][-1]

    # Choosing random seed and initial point
    seed = 522
    np.random.seed(seed)
    # x_initial = [np.ones(d) / np.sqrt(d), np.ones(d) / np.sqrt(d)] #changed here
    x_initial = np.ones(d)/np.sqrt(d) #changed here
    # Rand-1 sequence
    indices = np.random.randint(low=0, high=d, size=iterations)

    m = int(n/num_client)
    A_training_list = [A[i*m:(i+1)*m] for i in range(num_client)]
    L_list = []
    for i in range(num_client):
        L_i, lam_i_min, lam_i_max = cal_sum_rank_1(A_training_list[i])[0], cal_sum_rank_1(A_training_list[i])[1], cal_sum_rank_1(A_training_list[i])[2]
        L_i = L_i + 2 * lam * np.eye(d)  # Add the upperbound of the Hessian of the regularizer
        L_list.append(L_i)


    # Experiment 1
    if cur_exp == 1:
        ############################    Experiments 1: rand-1 case first four
        # Curve 1: Running FGD with scalar stepsize, scalar smoothness, rand-1
        arrnm_1 = fgd(0.5, s,beta, tune, nodes, weights, x_initial, max_eig_L, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        # Curve 2: Running FGD with scalar stepsize, matrix smoothness, rand-1
        arrnm_2 = fgd_mat(alpha, s,beta, tune, nodes, weights, x_initial, max_eig_L_diag, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        # Curve 3: Running CFGD with scalar stepsize, scalar smoothness, rand-1
        arrnm_3 = cfgd(alpha, s,beta, tune, nodes, weights, x_initial, max_eig_L, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        # Curve 4: Running CGD with scalar stepsize, matrix smoothness, rand-1
        arrnm_4 = cfgd_1(alpha, s,beta, tune, nodes, weights, x_initial, max_eig_L_diag, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        # Curve 5: Running CFGD with diagonal matrix stepsize, matrix smoothness, rand-1, algorithm 1
        arrnm_5 = cfgd_alg_1_L_diag_inv(alpha, s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        # Curve 6: Running CFGD with L-1 matrix stepsize, matrix smoothness, rand-1, algorithm 1
        arrnm_6 = cfgd_alg_1_L_inv(alpha, s,beta, tune, nodes, weights, x_initial, L, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        # Curve 7: Running CFGD with L-1/2 matrix stepsize, matrix smoothness, rand-1, algorithm 1
        arrnm_7 = cfgd_alg_1_L_half_inv(alpha, s,beta, tune, nodes, weights, x_initial, L, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        # Curve 8: Running CFGD with diagonal matrix stepsize, matrix smoothness, rand-1, algorithm 2
        arrnm_8 = cfgd_alg_2_L_diag_inv(alpha, s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        
    # Experiment 2
    if cur_exp == 2:
        # control the minibatch
        t1, t2, t3 = int(d / 4) + 1, int(d / 2) + 1, int(3 * d / 4) + 1 
        # t1, t2,t3 = 25, 30, 40    # Several t we want to examine
        ############################    Experiments 2: rand-t1 case
        # Curve 1: Running CFGD with scalar stepsize, scalar smoothness, rand-t1
        arrnm_1 = run_cfgd_curve_1_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, max_eig_L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t1)
        # Curve 2: Running CFGD with scalar stepsize, matrix smoothness, rand-t1
        arrnm_2 = run_cfgd_curve_2_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t1)
        # Curve 3: Running CGD alg1 with optimal matrix stepsize, matrix smoothness, rand-t1
        arrnm_3 = run_cfgd_curve_3_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t1)
        # Curve 4: Running CFGD alg2 with optimal matrix stepsize, matrix smoothness, rand-t1
        arrnm_4 = run_cfgd_curve_4_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t1)
        
        ############################    Experiments 2: rand-t2 case
        # Curve 1: Running CFGD with scalar stepsize, scalar smoothness, rand-t2
        arrnm_5 = run_cfgd_curve_1_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, max_eig_L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t2)
        # Curve 2: Running CFGD with scalar stepsize, matrix smoothness, rand-t2
        arrnm_6 = run_cfgd_curve_2_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t2)
        # Curve 3: Running CGD alg1 with optimal matrix stepsize, matrix smoothness, rand-t2
        arrnm_7 = run_cfgd_curve_3_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t2)
        # Curve 4: Running CFGD alg2 with optimal matrix stepsize, matrix smoothness, rand-t2
        arrnm_8 = run_cfgd_curve_4_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t2)
        
        ############################    Experiments 2: rand-t3 case
        # Curve 1: Running CFGD with scalar stepsize, scalar smoothness, rand-t3
        arrnm_9 = run_cfgd_curve_1_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, max_eig_L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t3)
        # Curve 2: Running CFGD with scalar stepsize, matrix smoothness, rand-t3
        arrnm_10 = run_cfgd_curve_2_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t3)
        # Curve 3: Running CGD alg1 with optimal matrix stepsize, matrix smoothness, rand-t3
        arrnm_11 = run_cfgd_curve_3_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t3)
        # Curve 4: Running CFGD alg2 with optimal matrix stepsize, matrix smoothness, rand-t3
        arrnm_12 = run_cfgd_curve_4_exp_2(alpha, s,beta, tune, nodes, weights, x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t3)


        if PLOT == True:
            iterations = np.arange(1, arrnm_1.shape[0] + 1)
            plt.xlabel("Iterations")
            plt.ylabel("Log average norm")
            plt.plot(iterations, np.log10(arrnm_1),
                     label="CGD, scalar stepsize, scalar smoothness, rand-{}, lam={}".format(t1, lam),
                     marker='o', markevery=250)
            plt.plot(iterations, np.log10(arrnm_2),
                     label="CGD, scalar stepsize, matrix smoothness, rand-{}, lam={}".format(t1, lam),
                     marker='v', markevery=250)
            plt.plot(iterations, np.log10(arrnm_3),
                     label="Alg2, optimal matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t1, lam),
                     marker='<', markevery=250)
            plt.plot(iterations, np.log10(arrnm_4),
                     label="Alg1, same matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t1, lam),
                     marker='^', markevery=250)
            plt.plot(iterations, np.log10(arrnm_5),
                     label="CGD, scalar stepsize, scalar smoothness, rand-{}, lam={}".format(t2, lam),
                     marker='1', markevery=250)
            plt.plot(iterations, np.log10(arrnm_6),
                     label="CGD, scalar stepsize, matrix smoothness, rand-{}, lam={}".format(t2, lam),
                     marker='2', markevery=250)
            plt.plot(iterations, np.log10(arrnm_7),
                     label="Alg2, optimal matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t2, lam),
                     marker='3', markevery=250)
            plt.plot(iterations, np.log10(arrnm_8),
                     label="Alg1, same matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t2, lam),
                     marker='4', markevery=250)
            plt.plot(iterations, np.log10(arrnm_9),
                     label="CGD, scalar stepsize, scalar smoothness, rand-{}, lam={}".format(t3, lam),
                     marker='H', markevery=250)
            plt.plot(iterations, np.log10(arrnm_10),
                     label="CGD, scalar stepsize, matrix smoothness, rand-{}, lam={}".format(t3, lam),
                     marker='+', markevery=250)
            plt.plot(iterations, np.log10(arrnm_11),
                     label="Alg2, optimal matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t3, lam),
                     marker='_', markevery=250)
            plt.plot(iterations, np.log10(arrnm_12),
                     label="Alg1, same matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t3, lam),
                     marker='8', markevery=250)
            plt.legend()
            plt.savefig(os.path.join("Temps" ,"Exp_2_" + dataset.split(sep='.')[0] +
                                     "_lam_{}_rand_{}".format(lam, t1) + ".png"))
            plt.show()

    # Experiment 3
    if cur_exp == 3:
        # Curve 1: uniform with scalar stepsize
        arrnm_1 = run_cgd_curve_1_exp_3(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)
        # Curve 2: uniform with scalar stepsize
        arrnm_2 = run_cgd_curve_2_exp_3(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)
        # Curve 3: algorithm 1 with uniform/importance with diagonal matrix stepsize
        # Note that the importance sampling probability is uniform in this case
        arrnm_3 = run_cgd_curve_3_exp_3(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)
        # Curve 5: algorithm 2 with uniform/importance with diagonal matrix stepsize
        # Note that the importance sampling probability is uniform in this case
        arrnm_5 = run_cgd_curve_5_exp_3(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)


        if PLOT == True:
            iterations = np.arange(1, arrnm_1.shape[0] + 1)
            plt.xlabel("Iterations")
            plt.ylabel("Log average norm")
            plt.plot(iterations, np.log10(arrnm_1),
                     label="CGD, scalar stepsize, matrix smoothness, rand-1-uniform, lam={}".format(lam),
                     marker='o', markevery=250)
            plt.plot(iterations, np.log10(arrnm_2),
                     label="CGD, scalar stepsize, matrix smoothness, rand-1-importance, lam={}".format(lam),
                     marker='1', markevery=250)
            plt.plot(iterations, np.log10(arrnm_3),
                     label="alg1, diagonal matrix stepsize, matrix smoothness, rand-1-imp/uniform, lam={}".format(lam),
                     marker='<', markevery=250)
            plt.plot(iterations, np.log10(arrnm_5),
                     label="alg2, diagonal matrix stepsize, matrix smoothness, rand-1-imp/uniform, lam={}".format(lam),
                     marker='v', markevery=200)
            plt.legend()
            plt.savefig(os.path.join("Temps" ,"Exp_3_" + dataset.split(sep='.')[0] +
                                     "_lam_{}_rand_1".format(lam) + ".png"))
            plt.show()

    # Experiment 4
    if cur_exp == 4:
        # Reshuffling the dataset
        # A, b = shuffle(A, b, random_state=0)
        # Splitting the dataset into smaller part
        split_A, split_b = np.array_split(A, num_client), np.array_split(b, num_client)

        # Calculating the smoothness matrix for function g alone
        L_g       = cal_sum_rank_1(A)[0]
        split_L_g = [cal_sum_rank_1(split_A[i])[0]  for i in range(num_client)]
        # Calculating the smoothness matrix for the whole function
        L         = L_g + 2 * lam * np.eye(d)
        split_L   = [split_L_g[i] + 2 * lam * np.eye(d) for i in range(num_client)]

        # Finding the maximum eigenvalue of the g part
        L_g_scalar = np.linalg.eigh(L_g)[0][-1]
        split_L_g_scalar = [np.linalg.eigh(split_L_g[i])[0][-1] for i in range(num_client)]

        # For convenience, we set lam = 0, now it becomes a convex objective to find the minimum
        # Finding minimum
        min_f = find_min_GD(A, b, L_g_scalar, lam=lam, plot=False, whole=True)
        split_min_f = [find_min_GD(split_A[i], split_b[i], split_L_g_scalar[i],
                                   lam=lam, plot=False, whole=False, index=i) for i in range(num_client)]

        # Setting the relative error level
        # epsilon_sq = 0.0001
        # Some values needed
        # Because we are using rand-1 sketch
        omega           = d - 1
        # Remember we have a regularizer in this case
        L_scalar        = L_g_scalar + 2 * lam
        # lam does not affect which one is the biggest
        L_max_scalar    = np.max(split_L_g_scalar) + 2 * lam
        # C should be larger than 0 （This is an upper bound on C）
        C = (2 / num_client) * (min_f - np.average(split_min_f))
        # assert C >= 0, "Assertion error, negative C={}".format(C)


        # For curve 1 & 2:
        upper_bound = [
            (1 / L_scalar),
            np.sqrt(num_client) / np.sqrt(omega * L_scalar * L_max_scalar * iterations),
            epsilon_sq / (2 * L_scalar * L_max_scalar * omega * C)
        ]
        # This is the largest gamma we can demand
        gamma = np.min(upper_bound)
        arrnm_1 = DCGD(x_initial, gamma, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)

        arrnm_2 = DCFGD(alpha, s, beta, tune, nodes, weights, x_initial, gamma, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)
        # For ______________________:
        # Upper bound 1
        upper_bound_1   = (1 / L_scalar)
        # Upper bound 2
        diff_L          = d * np.diag(np.diag(L)) - L
        split_L_half    = [sp.linalg.fractional_matrix_power(split_L[i], (1/2)) for i in range(num_client)]
        eigen_list      = [np.linalg.eigh(split_L_half[i] @ diff_L @ split_L_half[i])[0][-1] for i in range(num_client)]
        cond_2_max      = np.max(eigen_list)
        upper_bound_2   = np.sqrt(n) / np.sqrt(iterations * cond_2_max)
        # Upper bound 3
        cond_3_max     = cond_2_max
        upper_bound_3   = epsilon_sq / (2 * C * cond_3_max)
        # Merging upperbound
        upper_bound = [upper_bound_1, upper_bound_2, upper_bound_3]
        gamma = np.min(upper_bound)
        # arrnm_3 = DCGD_1(x_initial, gamma, num_client, iterations,
        #                                  lam=lam, plot=False, save=True, dataset=dataset)
        # arrnm_4 = DCFGD_1(x_initial, gamma, num_client, iterations,
        #                                  lam=lam, plot=False, save=True, dataset=dataset)
        arrnm_3 = DCGD_mat(x_initial, gamma, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)


        # For curve 3&4:
        L_diag          = np.diag(np.diag(L))
        L_diag_inv      = sp.linalg.inv(L_diag)
        L_diag_inv_half = sp.linalg.fractional_matrix_power(L_diag, -(1/2))
        # Upper bound 1
        upper_bound_1   = (1 / np.linalg.eigh(L_diag_inv_half @ L @ L_diag_inv_half)[0][-1])
        # Upper bound 2
        diff_L_alg_1 = d * L_diag_inv - L_diag_inv @ L @ L_diag_inv
        matrix_list_alg_1 = [split_L_half[i] @ diff_L_alg_1 @ split_L_half[i] for i in range(num_client)]
        max_list_eig_1 = np.max([np.linalg.eigh(matrix_list_alg_1[i])[0][-1] for i in range(num_client)])
        upper_bound_2 = np.sqrt(n) / np.sqrt(iterations * max_list_eig_1)
        # Upper bound 3
        L_diag_inv_1d = sp.linalg.fractional_matrix_power(L_diag_inv, 1/d)
        det_rhs       = sp.linalg.det(L_diag_inv_1d)
        upper_bound_3 = epsilon_sq / (2 * C * det_rhs)
        # Merging upperbound
        upper_bound = [upper_bound_1, upper_bound_2, upper_bound_3]
        gamma = np.min(upper_bound)
        # Getting stepsize
        D = gamma * L_diag_inv
        arrnm_4, arrstd_4 = DCGD_1(x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)

        arrnm_5, arrstd_5 = DCFGD_1(alpha, s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)

        # For curve 5&6: it is the same with curve 3 in this case
        arrnm_6, arrstd_6 = DCGD_2(x_initial, D, num_client, iterations, 
                                         lam=lam, plot=False, save=True, dataset=dataset)
        arrnm_7, arrstd_7 = DCFGD_2(alpha, s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)
        if PLOT == True:
            raise NotImplementedError
            iterations = np.arange(1, arrnm_1.shape[0] + 1)

    if cur_exp == 5:
        alpha = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        split_A, split_b = np.array_split(A, num_client), np.array_split(b, num_client)

        # Calculating the smoothness matrix for function g alone
        L_g       = cal_sum_rank_1(A)[0]
        split_L_g = [cal_sum_rank_1(split_A[i])[0]  for i in range(num_client)]
        # Calculating the smoothness matrix for the whole function
        L         = L_g + 2 * lam * np.eye(d)
        split_L   = [split_L_g[i] + 2 * lam * np.eye(d) for i in range(num_client)]

        # Finding the maximum eigenvalue of the g part
        L_g_scalar = np.linalg.eigh(L_g)[0][-1]
        split_L_g_scalar = [np.linalg.eigh(split_L_g[i])[0][-1] for i in range(num_client)]

        # For convenience, we set lam = 0, now it becomes a convex objective to find the minimum
        # Finding minimum
        min_f = find_min_GD(A, b, L_g_scalar, lam=lam, plot=False, whole=True)
        split_min_f = [find_min_GD(split_A[i], split_b[i], split_L_g_scalar[i],
                                   lam=lam, plot=False, whole=False, index=i) for i in range(num_client)]

        # Setting the relative error level
        # epsilon_sq = 0.0001
        # Some values needed
        # Because we are using rand-1 sketch
        omega           = d - 1
        # Remember we have a regularizer in this case
        L_scalar        = L_g_scalar + 2 * lam
        # lam does not affect which one is the biggest
        L_max_scalar    = np.max(split_L_g_scalar) + 2 * lam
        # C should be larger than 0 （This is an upper bound on C）
        C = (2 / num_client) * (min_f - np.average(split_min_f))
        # assert C >= 0, "Assertion error, negative C={}".format(C)
        split_L_half    = [sp.linalg.fractional_matrix_power(split_L[i], (1/2)) for i in range(num_client)]

        
        L_diag          = np.diag(np.diag(L))
        L_diag_inv      = sp.linalg.inv(L_diag)
        L_diag_inv_half = sp.linalg.fractional_matrix_power(L_diag, -(1/2))
        # Upper bound 1
        upper_bound_1   = (1 / np.linalg.eigh(L_diag_inv_half @ L @ L_diag_inv_half)[0][-1])
        # Upper bound 2
        diff_L_alg_1 = d * L_diag_inv - L_diag_inv @ L @ L_diag_inv
        matrix_list_alg_1 = [split_L_half[i] @ diff_L_alg_1 @ split_L_half[i] for i in range(num_client)]
        max_list_eig_1 = np.max([np.linalg.eigh(matrix_list_alg_1[i])[0][-1] for i in range(num_client)])
        upper_bound_2 = np.sqrt(n) / np.sqrt(iterations * max_list_eig_1)
        # Upper bound 3
        L_diag_inv_1d = sp.linalg.fractional_matrix_power(L_diag_inv, 1/d)
        det_rhs       = sp.linalg.det(L_diag_inv_1d)
        upper_bound_3 = epsilon_sq / (2 * C * det_rhs)
        # Merging upperbound
        upper_bound = [upper_bound_1, upper_bound_2, upper_bound_3]
        gamma = np.min(upper_bound)
        # Getting stepsize
        D = gamma * L_diag_inv

        arrnm_1, _ = DCFGD_2_ablation(alpha[0], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)   
        arrnm_2, _ = DCFGD_2_ablation(alpha[1], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset) 
        arrnm_3, _ = DCFGD_2_ablation(alpha[2], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)  
        arrnm_4, _ = DCFGD_2_ablation(alpha[3], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)
        arrnm_5, _ = DCFGD_2_ablation(alpha[4], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)  
        arrnm_6, _ = DCFGD_2_ablation(alpha[5], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)  
        arrnm_7, _ = DCFGD_2_ablation(alpha[6], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)  
        arrnm_8, _ = DCFGD_2_ablation(alpha[7], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)  
        arrnm_9, _ = DCFGD_2_ablation(alpha[8], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)  
        arrnm_10, _ = DCFGD_2_ablation(alpha[9], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)       
        arrnm_11, _ = DCFGD_2_ablation(alpha[10], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)  
        arrnm_12, _ = DCFGD_2_ablation(alpha[11], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset) 
        arrnm_13, _ = DCFGD_2_ablation(alpha[12], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)
        arrnm_14, _ = DCFGD_2_ablation(alpha[13], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)
        arrnm_15, _ = DCFGD_2_ablation(alpha[14], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)
        arrnm_16, _ = DCFGD_2_ablation(alpha[15], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)
        arrnm_17, _ = DCFGD_2_ablation(alpha[16], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)
        arrnm_18, _ = DCFGD_2_ablation(alpha[17], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)
        arrnm_19, _ = DCFGD_2_ablation(alpha[18], s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset) 

    if cur_exp == 6:
        alpha = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        arrnm_1 = cfgd_alg_2_L_diag_inv_ablation(alpha[0], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_2 = cfgd_alg_2_L_diag_inv_ablation(alpha[1], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_3 = cfgd_alg_2_L_diag_inv_ablation(alpha[2], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_4 = cfgd_alg_2_L_diag_inv_ablation(alpha[3], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_5 = cfgd_alg_2_L_diag_inv_ablation(alpha[4], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_6 = cfgd_alg_2_L_diag_inv_ablation(alpha[5], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_7 = cfgd_alg_2_L_diag_inv_ablation(alpha[6], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_8 = cfgd_alg_2_L_diag_inv_ablation(alpha[7], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_9 = cfgd_alg_2_L_diag_inv_ablation(alpha[8], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_10 = cfgd_alg_2_L_diag_inv_ablation(alpha[9], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_11 = cfgd_alg_2_L_diag_inv_ablation(alpha[10], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_12 = cfgd_alg_2_L_diag_inv_ablation(alpha[11], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_13 = cfgd_alg_2_L_diag_inv_ablation(alpha[12], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_14 = cfgd_alg_2_L_diag_inv_ablation(alpha[13], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_15 = cfgd_alg_2_L_diag_inv_ablation(alpha[14], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_16 = cfgd_alg_2_L_diag_inv_ablation(alpha[15], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_17 = cfgd_alg_2_L_diag_inv_ablation(alpha[16], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_18 = cfgd_alg_2_L_diag_inv_ablation(alpha[17], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)
        arrnm_19 = cfgd_alg_2_L_diag_inv_ablation(alpha[18], s,beta, tune, nodes, weights, x_initial, L_diag_inv, A, b, iterations, lam, plot=False, save=True, dataset=dataset)        


    if cur_exp == 7:
        # Reshuffling the dataset
        # A, b = shuffle(A, b, random_state=0)
        # Splitting the dataset into smaller part
        split_A, split_b = np.array_split(A, num_client), np.array_split(b, num_client)

        # Calculating the smoothness matrix for function g alone
        L_g       = cal_sum_rank_1(A)[0]
        split_L_g = [cal_sum_rank_1(split_A[i])[0]  for i in range(num_client)]
        # Calculating the smoothness matrix for the whole function
        L         = L_g + 2 * lam * np.eye(d)
        split_L   = [split_L_g[i] + 2 * lam * np.eye(d) for i in range(num_client)]

        # Finding the maximum eigenvalue of the g part
        L_g_scalar = np.linalg.eigh(L_g)[0][-1]
        split_L_g_scalar = [np.linalg.eigh(split_L_g[i])[0][-1] for i in range(num_client)]

        # For convenience, we set lam = 0, now it becomes a convex objective to find the minimum
        # Finding minimum
        min_f = find_min_GD(A, b, L_g_scalar, lam=lam, plot=False, whole=True)
        split_min_f = [find_min_GD(split_A[i], split_b[i], split_L_g_scalar[i],
                                   lam=lam, plot=False, whole=False, index=i) for i in range(num_client)]

        # Setting the relative error level
        # epsilon_sq = 0.0001
        # Some values needed
        # Because we are using rand-1 sketch
        omega           = d - 1
        # Remember we have a regularizer in this case
        L_scalar        = L_g_scalar + 2 * lam
        # lam does not affect which one is the biggest
        L_max_scalar    = np.max(split_L_g_scalar) + 2 * lam
        # C should be larger than 0 （This is an upper bound on C）
        C = (2 / num_client) * (min_f - np.average(split_min_f))
        # assert C >= 0, "Assertion error, negative C={}".format(C)


        # For curve 1 & 2:
        upper_bound = [
            (1 / L_scalar),
            np.sqrt(num_client) / np.sqrt(omega * L_scalar * L_max_scalar * iterations),
            epsilon_sq / (2 * L_scalar * L_max_scalar * omega * C)
        ]
        # This is the largest gamma we can demand
        gamma = np.min(upper_bound)
        # arrnm_1 = DCGD(x_initial, gamma, num_client, iterations, <-- UNCOMMENT IT
        #                                  lam=lam, plot=False, save=True, dataset=dataset)

        # arrnm_2 = DCFGD(alpha, s, beta, tune, nodes, weights, x_initial, gamma, num_client, iterations,
        #                                  lam=lam, plot=False, save=True, dataset=dataset)
        # For ______________________:
        # Upper bound 1
        upper_bound_1   = (1 / L_scalar)
        # Upper bound 2
        diff_L          = d * np.diag(np.diag(L)) - L
        split_L_half    = [sp.linalg.fractional_matrix_power(split_L[i], (1/2)) for i in range(num_client)]
        eigen_list      = [np.linalg.eigh(split_L_half[i] @ diff_L @ split_L_half[i])[0][-1] for i in range(num_client)]
        cond_2_max      = np.max(eigen_list)
        upper_bound_2   = np.sqrt(n) / np.sqrt(iterations * cond_2_max)
        # Upper bound 3
        cond_3_max     = cond_2_max
        upper_bound_3   = epsilon_sq / (2 * C * cond_3_max)
        # Merging upperbound
        upper_bound = [upper_bound_1, upper_bound_2, upper_bound_3]
        gamma = np.min(upper_bound)

        # For curve 3&4:
        L_diag          = np.diag(np.diag(L))
        L_diag_inv      = sp.linalg.inv(L_diag)
        L_diag_inv_half = sp.linalg.fractional_matrix_power(L_diag, -(1/2))
        # Upper bound 1
        upper_bound_1   = (1 / np.linalg.eigh(L_diag_inv_half @ L @ L_diag_inv_half)[0][-1])
        # Upper bound 2
        diff_L_alg_1 = d * L_diag_inv - L_diag_inv @ L @ L_diag_inv
        matrix_list_alg_1 = [split_L_half[i] @ diff_L_alg_1 @ split_L_half[i] for i in range(num_client)]
        max_list_eig_1 = np.max([np.linalg.eigh(matrix_list_alg_1[i])[0][-1] for i in range(num_client)])
        upper_bound_2 = np.sqrt(n) / np.sqrt(iterations * max_list_eig_1)
        # Upper bound 3
        L_diag_inv_1d = sp.linalg.fractional_matrix_power(L_diag_inv, 1/d)
        det_rhs       = sp.linalg.det(L_diag_inv_1d)
        upper_bound_3 = epsilon_sq / (2 * C * det_rhs)
        # Merging upperbound
        upper_bound = [upper_bound_1, upper_bound_2, upper_bound_3]
        gamma = np.min(upper_bound)
        # Getting stepsize
        D = gamma * L_diag_inv
    
        alpha_range = np.arange(0, 1.05, 0.05)
        beta_range = np.arange(0, 10.5, 0.5)
        tune_range = np.arange(-10, 10.05, 0.05)
        a_range = np.arange(0.05, 1, 0.05)
        bb_range = np.arange(0.05, 1, 0.05)

# Generate all possible combinations
        param_combinations = list(itertools.product(alpha_range, beta_range, tune_range, a_range, bb_range))

# Initialize variables to track the minimum value and corresponding parameters
        min_value = float('inf')
        best_params = None

# Iterate over each combination and call the function
        for alpha, beta, tune, a, b_ in tqdm(param_combinations, desc="Tuning Hyperparameters"):
            s=1
            nodes, weights = gauss_jacobi_quadrature(s,a,b_)
            arrnm_7, arrstd_7 = DCFGD_2(alpha, s, beta, tune, nodes, weights, x_initial, D, num_client, iterations,
                                                lam=lam, plot=False, save=True, dataset=dataset)  
    
    # Check if the last value of arrnm_7 is the smallest we've seen
            if arrnm_7[-1] < min_value:
                min_value = arrnm_7[-1]
                best_params = (alpha, beta, tune, nodes, weights)

# Print the best parameters and the corresponding minimum value
        print(f"The set of parameters for which the last value of arrnm_7 is minimum:")
        print(f"Alpha: {best_params[0]}, Beta: {best_params[1]}, Tune: {best_params[2]}, Nodes: {best_params[3]}, Weights: {best_params[4]}")
        print(f"Minimum last value of arrnm_7: {min_value}")
                  

    if cur_exp == 8:
        split_A, split_b = np.array_split(A, num_client), np.array_split(b, num_client)
        arrnm_marina = det_CGD2_VR(x_initial, L, num_client, iterations=10000, lam=0.1,p=0.2,t=1)








