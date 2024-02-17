from fenics import *
from pyfftw import *
from numpy.random import default_rng
import numpy as np
import math
import time
import logging
# from tqdm import tqdm

import cov_functions
import periodisation_smooth
import estimate_samples_LD

# suppress information messages (printing takes more time)
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.WARNING)

# generator for random variables
rng = default_rng()

def MLMC_simulation(x, y, a, epsilon, alpha, C_alpha, C_tilde_alpha, gamma, C_gamma, m_0, cov_fun, cov_fun_per, sigma=1, rho=0.3, nu=0.5, p=1, pol_degree=1):
    '''
    Runs one simulation of MLMC with smooting for computing E[p(x,y)]. In particular, implements MLMC algorithm in Cliffe (2011).

    :param x: the x value at which to compute E[p(x,y)].
    :param y: the y value at which to compute E[p(x,y)].
    :param a: the RHS constant.
    :param epsilon: the required accuracy.
    :param alpha: the value of alpha required for Richardson extrapolation.
    :param C_alpha: the constant in |E[Q_l - Q_{l-1}]| = C_\alpha h^\alpha. 
    :param C_tilde_alpha: the constant in |E[Q_l - \tilde{Q}_{l-1}]| = \tilde{C}_\alpha h^\alpha. 
    :param gamma: the slope of the cost required per simulation.
    :param C_gamma: the constant in Cost = C * M ^ gamma.
    :param m_0 (default 2): the mesh size on the coarsest level.
    :param cov_fun: covariance function.
    :param cov_fun_per: periodisation function to be used.
    :param sigma (default 1): variance of the covariance function.
    :param rho (default 0.3): correlation length of the covariance function.
    :param nu: smoothness of the covariance function.
    :param p: norm of covariance function argument.
    :param pol_degree (default 1): the degree of the polynomials used in FEM approximation.

    :return:
        rmse - the final RMSE. 
        total_time - time taken for one simulation.
    '''

    # different mesh sizes used on each level
    Ms = np.array([(m_0*(2**l))**2 for l in range(7)])
    # the cost per level
    C_ls = np.array([(np.exp(C_gamma))*(Ms[l]**gamma) + (np.exp(C_gamma))*(Ms[l-1]**gamma) for l in range(1,len(Ms))])
    C_ls = np.insert(C_ls, 0, (np.exp(C_gamma))*(Ms[0]**gamma))
    Ns = [5000, 4000, 3000, 1500, 750, 375, 187]
    # arrays for saving quantities of interest
    V_ls_NS = []
    V_ls_S = []
    exp_ls_NS = []
    exp_ls_S = []
    Y_hats_NS = []
    Y_hats_S = []

    ##### STEP 1 - Start with L = 0
    L = 0

    total_time_t0 = time.time()

    while(True):
        ##### STEP 2 - Estimate variance using initial number of samples
        Y_hat_S, Y_hat_var_S, Y_l_S = \
            estimate_samples_LD.Y_l_smoothed_est_fun(Ns[L], x, y, a, L, alpha, C_alpha, C_tilde_alpha, m_0, cov_fun, cov_fun_per, sigma, rho, nu, p, pol_degree)
        Y_hat_NS, Y_hat_var_NS, Y_l_NS = \
            estimate_samples_LD.Y_l_not_smoothed_est_fun(Ns[L], x, y, a, L, alpha, C_alpha, C_tilde_alpha, m_0, cov_fun, cov_fun_per, sigma, rho, nu, p, pol_degree)

        # save initial variance, expectation and samples computed
        V_ls_S.append(Y_hat_var_S)
        exp_ls_S.append(Y_hat_S)
        Y_hats_S.append(Y_l_S)
        V_ls_NS.append(Y_hat_var_NS)
        exp_ls_NS.append(Y_hat_NS)
        Y_hats_NS.append(Y_l_NS)
        
        ##### STEP 3 - Calculate optimal number of samples
        N_ls = np.array([math.ceil((np.sum(np.sqrt(C_ls[:L+1]*V_ls_S[:L+1]))+np.sqrt(C_ls[L]*V_ls_NS[L])) * 2 * epsilon ** (-2) * np.sqrt(V_ls_S[l] / C_ls[l])) for l in range(L+1)])

        N_ls = np.append(N_ls, math.ceil((np.sum(np.sqrt(C_ls[:L+1]*V_ls_S[:L+1]))+np.sqrt(C_ls[L]*V_ls_NS[L]))* 2 * epsilon ** (-2) * np.sqrt(V_ls_NS[L] / C_ls[L])))

        N_ls = N_ls.astype(int)

        ##### STEP 4 - Evaluate extra samples at each level
        # smoothed levels
        for l in range(L):
            Y_hat_S_l = Y_hats_S[l]
            N_old = len(Y_hat_S_l)

            # if we need more samples
            if(N_ls[l] > N_old):
                Y_hat_S_l = np.resize(Y_hat_S_l, N_ls[l])
                N_new_samples = N_ls[l] - N_old

                # compute extra sampled needed
                _, _, Y_hat_S_extra = estimate_samples_LD.Y_l_smoothed_est_fun(N_new_samples, x, y, a, L, alpha, C_alpha, C_tilde_alpha, m_0, cov_fun, cov_fun_per, sigma, rho, nu, p, pol_degree)
                    
                # save extra samples to re-use
                Y_hat_S_l[N_old:] = Y_hat_S_extra

                # Compute new expectation and variance and save in array
                exp_S_l = np.average(Y_hat_S_l)
                exp_ls_S[l] = exp_S_l

                var_S_l = 1/(N_ls[l]-1) * np.sum((Y_hat_S_l - exp_ls_S[l]) ** 2)
                V_ls_S[l] = var_S_l
                Y_hats_S[l] = Y_hat_S_l

        # not smoothed level levels
        Y_hat_NS_l = Y_hats_NS[L]
        N_old = len(Y_hat_NS_l)

        # if we need more samples
        if(N_ls[L] > N_old):
            Y_hat_NS_l = np.resize(Y_hat_NS_l, N_ls[L])
            N_new_samples = N_ls[L] - N_old

            # compute extra sampled needed
            _, _, Y_hat_NS_extra = estimate_samples_LD.Y_l_not_smoothed_est_fun(N_new_samples, x, y, a, L, alpha, C_alpha, C_tilde_alpha, m_0, cov_fun, cov_fun_per, sigma, rho, nu, p, pol_degree)
                    
            # save extra samples to re-use
            Y_hat_NS_l[N_old:] = Y_hat_NS_extra

            # Compute new expectation and variance and save in array
            exp_NS_l = np.average(Y_hat_NS_l)
            exp_ls_NS[L] = exp_NS_l

            var_NS_l = 1/(N_ls[L]-1) * np.sum((Y_hat_NS_l - exp_ls_NS[L]) ** 2)
            V_ls_NS[L] = var_NS_l
            Y_hats_NS[L] = Y_hat_NS_l

        # ### STEP 5 - Test for convergence if L >= 1

        # Compute sample variance across all levels
        sample_var = np.sum(1 / N_ls[:L] * V_ls_S[:L]) + 1/N_ls[L] * V_ls_NS[L]

        # only test for converges if we have at least 2 levels (0 and 1)
        if (L >= 1):
            disc_error = (exp_ls_NS[L] / (1-C_tilde_alpha/C_alpha*4 **alpha))**2

            # compute RMSE (for refernce)
            rmse = np.sqrt(sample_var + disc_error)  

        ### STEP 6 - If not converged, set L = L+1
        if (L == 0):
            L += 1  
        elif (disc_error > epsilon**2/2):
            L += 1
        else:
            # break the loop once we have converged
            break

    # compute final expectation estimate
    exp_est = np.sum(exp_ls_S[:L]) + exp_ls_NS[L]
    # compute total time taken
    total_time_t1 = time.time()
    total_time = total_time_t1-total_time_t0

    return rmse, total_time

# below is an example of running a Multilevel Monte Carlo with smoothing simulation for a given accuracy
# uncomment to run
# if __name__ == "__main__":
#     # the value at which to compute E[p(x,y)]
#     x_val = 7 / 15
#     y_val = 7 / 15
#     # choose RHS constant for the ODE
#     a_val = 1
#     # polynomial degree for computing FEM approximations
#     pol_degree_val = 1
#     # variance of random field
#     sigma = 1
#     # correlation length of random field
#     rho = 0.03
#     # smoothness parameter
#     nu = 1.5
#     # mesh size for coarsest level
#     m_0_val = 4
#     # norm to be used in covariance function of random field
#     p_val = 2

#     # these are the pre-computed values which you can use
#     alpha_val = 0.3203869583916329
#     C_alpha = -3.631204069258538
#     C_tilde_alpha = -4.125424298666542
#     beta_val = 1.325511867482945
#     C_beta = -0.935727804503858
#     gamma_val = 0.8700621125059221
#     C_gamma = -7.510443908031861

#     # list of different accuracies
#     epsilon = 0.1

#     # MLMC simulation
#     rmse_val, total_time_val = MLMC_simulation(x_val, y_val, a_val, epsilon, alpha_val, C_alpha, C_tilde_alpha, gamma_val, C_gamma, m_0_val, cov_functions.Matern_cov, periodisation_smooth.periodic_cov_fun, sigma, rho, nu, p_val, pol_degree_val)

#     print(f'Total time is {total_time_val}.')
#     print(f'Final RMSE is {rmse_val} < {epsilon}.\n\n')