#!/work/sc073/sc073/s2079009/miniconda3/bin/python

from fenics import *
from pyfftw import *
from numpy.random import default_rng
import numpy as np
import logging
import sys
sys.path.insert(0, '/home/s2079009/MAC-MIGS/PhD/PhD_code/')
from utils import cov_functions
from MLMCwLevDepCE import estimate_samples_LD

# suppress information messages (printing takes more time)
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.WARNING)

# generator for random variables
rng = default_rng()

def compute_alpha_beta(J1, J2, cov_fun, x, y, a, m_0, pol_degree=1, sigma=1, rho=0.3, nu=0.5, p=1):
    '''
    Computes alpha in E[Q_M - Q] = C * M ^ (-alpha), beta in V[Q_M-Q] = C * M ^ (-beta) and gamma in Cost = C * M ^ gamma by computing approximations on different levels. Uses Y_l_est_fun.

    :param x: the x point at which to compute E[p(x,y)].
    :param y: the y point at which to compute E[p(x,y)].
    :param a: the RHS constant of the PDE.
    :param m_0: mesh size on level zero.
    :param pol_degree (default 1): the degree of the polynomials used in FEM approximation.
    :param rho (default 0.3): correlation length of the covariance function.
    :param sigma (default 1): variance of the covariance function.

    :return: 
        Ms - different grid sizes used in computing approximations.
        Y_hats - E[Q_l-Q_{l-1}] for different levels l.
        alpha - slope of Y_hats line on log-log scale. 
        C_alpha - constant in E[Q_M - Q] = C * M ^ (-alpha).
        Y_hat_vars - V[Q_l-Q_{l-1}] for different levels l.
        beta - slope of Y_hat_vars line on log-log scale.
        C_beta - contant in V[Q_M-Q] = C * M ^ (-beta).
        Y_ls - samples of Q_l - Q_l-1 used in approximation.
        avg_times - ndarray of time taken on average for one simulation.
        gamma - the slope on the log-log scale of times.
        C_gamma -  the constant in Cost = C * M^gamma.
    '''

    # empty lists for storing the approximations and grid size.
    Y_hats = np.zeros(7)
    Y_hat_vars = np.zeros(7)
    Ms = np.zeros(7)
    Y_ls = []

    # number of samples needed to compute initial estimate of sample variance on each level
    Ns = [10000, 5000, 2500, 1200, 500, 250, 100]

    # use 7 levels - from 0 to 6
    for l in range(7):
        # for each level, compute expectation and variance
        Y_hat, Y_hat_var, Y_l = \
            estimate_samples_LD.Y_l_not_smoothed_est_fun(Ns[l], J1, J2, cov_fun, x, y, a, l, m_0, pol_degree, sigma, rho, nu, p)

        # save current values
        Ms[l] = (m_0 * 2**l) ** 2
        Y_hats[l] = np.abs(Y_hat)   
        Y_hat_vars[l] = Y_hat_var
        Y_ls.append(Y_l)

    # compute alpha and C_alpha by finding best linear fit through Y_hats
    alpha, C_alpha = np.polyfit(x=np.log(Ms[1:]), y=np.log(Y_hats[1:]), deg=1)
    # compute beta and C_beta by finding best linear fit through Y_hat_vars
    beta, C_beta = np.polyfit(x=np.log(Ms[1:]), y=np.log(Y_hat_vars[1:]), deg=1)

    return Ms, Y_hats, -alpha, C_alpha, Y_hat_vars, -beta, C_beta, Y_ls

def main():
    # the value at which to compute E[p(x,y)]
    x_val = 7 / 15
    y_val = 7 / 15
    # choose RHS constant for the ODE
    a_val = 1
    # polynomial degree for computing FEM approximations
    pol_degree_val = 1
    # variance of random field
    sigma = 1
    # correlation length of random field
    rho = 0.3
    # smoothness parameter
    nu = 0.5
    # mesh size for coarsest level
    m_0_val = 4
    # initial padding in circulant embedding matrix
    J1_val = J2_val = 0
    # norm to be used in covariance function of random field
    p_val = 1

    M_vec, Y_hat_vec, alpha_val, C_alpha, Y_hat_var_vec, beta_val, C_beta, Y_ls_vec = compute_alpha_beta(J1=J1_val, J2=J2_val, cov_fun=cov_functions.expopnential_cov, x=x_val, y=y_val, a=a_val, m_0=m_0_val, pol_degree=pol_degree_val, sigma=sigma, rho=rho, nu=nu, p=p_val)

    print(f'alpha_val={alpha_val}')
    print(f'C_alpha={C_alpha}')
    print(f'beta_val={beta_val}')
    print(f'C_beta={C_beta}')

    np.save('./data/Y_hat_vec.npy', Y_hat_vec, allow_pickle=True)
    np.save('./data/Y_hat_var_vec.npy', Y_hat_var_vec, allow_pickle=True)
    np.save('./data/Y_ls_vec.npy', Y_ls_vec, allow_pickle=True)

if __name__ == "__main__":
    main()