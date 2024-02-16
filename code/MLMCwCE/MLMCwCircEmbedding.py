from fenics import *
from pyfftw import *
from numpy.random import default_rng
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import logging
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/s2079009/MAC-MIGS/PhD/PhD_code/')
from MLMCwCE import estimate_samples
from utils import cov_functions
from utils import periodisation_smooth

# suppress information messages (printing takes more time)
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.WARNING)

# generator for random variables
rng = default_rng()

def MLMC_simulation(Y_ls, Y_hat, Y_var, x, y, a, epsilon, alpha, gamma, C_gamma, time_per_sim, cov_fun, cov_fun_per, m_0=4, pol_degree=1, sigma=1, rho=0.3, nu=0.5, p=1):
    '''
    Runs one simulation of MLMC for computing E[p(x,y)]. In particular, implements MLMC algorithm in Cliffe (2011).

    :param x: the x value at which to compute E[p(x,y)].
    :param y: the y value at which to compute E[p(x,y)].
    :param a: the RHS constant.
    :param epsilon: the required accuracy.
    :param Y_hat: previously computed approximations of E[Y_l].
    :param Y_hat_var: previously computed approximations of V[Y_l].
    :param alpha: the value of alpha required for Richardson extrapolation.
    :param gamma: the slope of the cost required per simulation.
    :param C_gamma: the constant in Cost = C * M ^ gamma.
    :param time_per_sim: ndarray with time taken to compute approximation on each grid with mesh size from 2^1 to 2^8.
    :param m_KL: number of modes to be included in KL-expansion.
    :param m_0 (default 2): the mesh size on the coarsest level.
    :param pol_degree (default 1): the degree of the polynomials used in FEM approximation.
    :param rho (default 0.3): correlation length of the covariance function.
    :param sigma (default 1): variance of the covariance function.

    :return:
        rmse - the final RMSE. 
        total_time - time taken for one simulation.
        exp_est - the final expectation estimate.
        N_ls - ndarray with number of samples needed for each level.
    '''

    # different mesh sizes used on each level
    Ms = np.array([(m_0*(2**l))**2 for l in range(7)])
    # the cost per level
    C_ls = np.array([(np.exp(C_gamma))*(Ms[l]**gamma) + (np.exp(C_gamma))*(Ms[l-1]**gamma) for l in range(1,7)])
    C_ls = np.insert(C_ls, 0, (np.exp(C_gamma))*(Ms[0]**gamma))
    # Ns = [3000, 1500, 750, 375, 187]
    # arrays for saving quantities of interest
    V_ls = Y_var
    exp_ls = Y_hat
    Y_hats = Y_ls

    ##### STEP 1 - Start with L = 0
    L = 0

    total_time_t0 = time.time()

    while(True):
        print(f'\n\nStarting level {L}.\n')

        ##### STEP 2 - Estimate variance using initial number of samples
        
        # print(f'Time estimated is {Ns[L]*time_per_sim[L] + (Ns[L]*time_per_sim[L-1] if (L!=0) else 0)} sec.')

        # t0 = time.time()
        # Y_hat, Y_hat_var, Y_l = \
        #     Y_l_est_fun(Ns[L],egnv,d,x, y, a, L, m_0, pol_degree, rho, sigma)
        # t1 = time.time()
        # print(f'Time taken is {t1-t0} seconds.')

        # # save initial variance, expectation and samples computed
        # V_ls[L] = Y_hat_var
        # exp_ls[L] = Y_hat
        # Y_hats.append(Y_l)

        print(f"Expectation estimate is {np.sum(exp_ls[:L+1])}.")

        ##### STEP 3 - Calculate optimal number of samples
        N_ls = np.array([ math.ceil(\
            np.sum( np.sqrt( C_ls[:L+1] * V_ls[:L+1] ) ) \
                * 2 * epsilon ** (-2) * np.sqrt( V_ls[l] / C_ls[l] ) ) \
                    for l in range(L+1)])

        print(f'New number of samples is {N_ls}.\n')

        ##### STEP 4 - Evaluate extra samples at each level
        for l in range(L+1):

            Y_hat_l = Y_hats[l]
            N_old = len(Y_hat_l)

            # if we need more samples
            if(N_ls[l] > N_old):
                Y_hat_l = np.resize(Y_hat_l, N_ls[l])
                N_new_samples = N_ls[l] - N_old

                print(f'New samples needed on level {l} is {N_new_samples}.')
                print(f'Time estimated is {N_new_samples*time_per_sim[l] + (N_new_samples*time_per_sim[l-1] if (l!=0) else 0)} sec.')

                t0 = time.time()

                # compute extra sampled needed
                _, _, Y_hat_extra = estimate_samples.Y_l_est_fun(N_new_samples, x, y, a, l, m_0, cov_fun, cov_fun_per, pol_degree, sigma, rho, nu, p)

                t1 = time.time()
                print(f'Time taken is {t1-t0} sec.')
                
                # save extra samples to re-use
                Y_hat_l[N_old:] = Y_hat_extra

                # Compute new expectation and variance and save in array
                exp_l = np.average(Y_hat_l)
                var_l = 1 / (N_ls[l]-1) * np.sum((Y_hat_l - exp_ls[l]) ** 2)

                V_ls[l] = var_l
                exp_ls[l] = exp_l
                Y_hats[l] = Y_hat_l

                print(f"New expectation estimate is {np.sum(exp_ls[:L+1])}. \n")

        ### STEP 5 - Test for convergence if L >= 1

        # Compute sample variance across all levels
        sample_var = np.sum(1 / N_ls * V_ls[:L+1])
        print(f'The sample variance is {sample_var} < {epsilon**2/2}.')

        # only test for converges if we have at least 2 levels (0 and 1)
        if (L >= 1):
            # compute discretisation error
            disc_error = (exp_ls[L] / (1-4 ** alpha))**2
            print(f'Discretisation error is {disc_error} < {epsilon**2/2}.')

            # compute RMSE (for refernce)
            rmse = np.sqrt(sample_var + disc_error)
            print(f'RMSE is {rmse} < {epsilon}.\n')    

        ### STEP 6 - If not converged, set L = L+1
        if (L == 0):
            L += 1  
        elif (disc_error > epsilon**2/2):
            L += 1
        else:
            # break the loop once we have converged
            print(f'Stopped at level {L}.\n')
            break

    # compute final expectation estimate
    exp_est = np.sum(exp_ls[:L+1])
    # compute total time taken
    total_time_t1 = time.time()
    total_time = total_time_t1-total_time_t0

    return Y_hats, exp_ls, V_ls, rmse, total_time, exp_est, N_ls

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
    rho = 0.03
    # smoothness parameters
    nu = 1.5
    # mesh size for coarsest level
    m_0_val = 16
    # norm to be used in covariance function of random field
    p_val = 2

    Y_hat_vec = np.load('./data/Y_hat_vec_003_15_norm.npy', allow_pickle=True)
    alpha_val = 0.3551417357831946
    C_alpha = -3.631204069258538
    Y_hat_var_vec = np.load('./data/Y_hat_var_vec_003_15_norm.npy', allow_pickle=True)
    beta_val = 1.5666865274225823
    C_beta = 1.470065536870814
    Y_ls_vec = np.load('./data/Y_ls_vec_003_15_norm.npy', allow_pickle=True)
    avg_times_vec = np.load('../MLMCwLevDepCE/data/avg_times_vec_LD_15.npy')
    avg_times_vec = avg_times_vec[3:]
    gamma_val = 0.9868719259553745
    C_gamma = -8.510608041617592

    epsilon = 0.00075 # accuracy

    print(f'Starting simulation for epsilon = {epsilon}.')

    # MLMC simulation
    Y_ls_vec, Y_hat_vec, Y_hat_var_vec, rmse_val, total_time_val, exp_est_val, N_ls_val = MLMC_simulation(Y_ls_vec, Y_hat_vec, Y_hat_var_vec, x_val, y_val, a_val, epsilon, alpha_val, gamma_val, C_gamma, avg_times_vec, cov_functions.Matern_cov, periodisation_smooth.periodic_cov_fun, m_0_val, pol_degree_val, sigma, rho, nu, p_val)

    print(f'Final estimate is {exp_est_val}.')
    print(f'Total time is {total_time_val}.')
    print(f'Final RMSE is {rmse_val} < {epsilon}.\n\n')

    # save current values
    print(f'rmse_val = {rmse_val}')
    print(f'total_time_val = {total_time_val}')
    print(f'exp_est_val = {exp_est_val}')
    print(f'N_ls_val = {N_ls_val}')

if __name__ == "__main__":
    main()