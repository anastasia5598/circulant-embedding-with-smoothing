from fenics import *
from pyfftw import *
from numpy.random import default_rng
import numpy as np
import time
import logging
from tqdm import tqdm
import math
import sys
sys.path.insert(0, '/home/s2079009/MAC-MIGS/PhD/PhD_code/')
from utils import cov_functions
from utils import periodisation_smooth
from MLMCwLevDepCE import estimate_samples_LD
from MLMCwCE import estimate_samples

# suppress information messages (printing takes more time)
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.WARNING)

# generator for random variables
rng = default_rng()

def MLMC_simulation_2(Y_ls_S, Y_ls_NS, Y_ls_mlmc, Y_hat_S, Y_var_S, Y_hat_NS, Y_var_NS, Y_hat_mlmc, Y_var_mlmc, smoothing_level, x, y, a, epsilon, alpha, C_alpha, C_tilde_alpha, gamma, C_gamma, time_per_sim, cov_fun, cov_fun_per, m_0=4, pol_degree=1, sigma=1, rho=0.3, nu=0.5, p=1):
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
    C_ls = np.array([(np.exp(C_gamma))*(Ms[l]**gamma) + (np.exp(C_gamma))*(Ms[l-1]**gamma) for l in range(1,len(Ms))])
    C_ls = np.insert(C_ls, 0, (np.exp(C_gamma))*(Ms[0]**gamma))
    # arrays for saving quantities of interest
    V_ls_NS = Y_var_NS
    V_ls_S = Y_var_S
    V_ls_mlmc = np.zeros(len(V_ls_S))
    V_ls_mlmc[2:] = Y_var_mlmc
    exp_ls_NS = Y_hat_NS
    exp_ls_S = Y_hat_S
    exp_ls_mlmc = np.zeros(len(exp_ls_S))
    exp_ls_mlmc[2:] = Y_hat_mlmc
    Y_hats_NS = Y_ls_NS
    Y_hats_S = Y_ls_S
    Y_hats_mlmc = Y_ls_mlmc
    Y_hats_mlmc = np.insert(Y_hats_mlmc, 0, np.zeros(2))

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

        print(f"Expectation estimate is {np.sum(exp_ls_S[:L])+exp_ls_NS[L]}.")
        
        ##### STEP 3 - Calculate optimal number of samples
        if (L <= smoothing_level+1):
            # in this case, we smooth all the levels, and only the last level has to not be smoothed
            N_ls = np.array([math.ceil((np.sum(np.sqrt(C_ls[:L]*V_ls_S[:L]))+np.sqrt(C_ls[L]*V_ls_NS[L])) * 2 * epsilon ** (-2) * np.sqrt(V_ls_S[l] / C_ls[l])) for l in range(L)])

            N_ls = np.append(N_ls, math.ceil((np.sum(np.sqrt(C_ls[:L]*V_ls_S[:L]))+np.sqrt(C_ls[L]*V_ls_NS[L]))* 2 * epsilon ** (-2) * np.sqrt(V_ls_NS[L] / C_ls[L])))
        else:
            # in this case, we smooth all the levels for which l <= ls, we do not smooth the ls+1 level (but ls is smoothed, so we need to take this into account), and use not smoothed differences for all the other levels
            N_ls = np.array([math.ceil((np.sum(np.sqrt(C_ls[:smoothing_level+1]*V_ls_S[:smoothing_level+1]))+np.sqrt(C_ls[smoothing_level+1]*V_ls_NS[smoothing_level+1])+np.sum(np.sqrt(C_ls[smoothing_level+2:L+1]*V_ls_mlmc[smoothing_level+2:L+1]))) * 2 * epsilon ** (-2) * np.sqrt(V_ls_S[l] / C_ls[l])) for l in range(smoothing_level+1)])

            N_ls = np.append(N_ls, math.ceil((np.sum(np.sqrt(C_ls[:smoothing_level+1]*V_ls_S[:smoothing_level+1]))+np.sqrt(C_ls[smoothing_level+1]*V_ls_NS[smoothing_level+1])+np.sum(np.sqrt(C_ls[smoothing_level+2:L+1]*V_ls_mlmc[smoothing_level+2:L+1]))) * 2 * epsilon ** (-2) * np.sqrt(V_ls_NS[smoothing_level+1] / C_ls[smoothing_level+1])))

            N_ls = np.concatenate((N_ls, np.array([math.ceil((np.sum(np.sqrt(C_ls[:smoothing_level+1]*V_ls_S[:smoothing_level+1]))+np.sqrt(C_ls[smoothing_level+1]*V_ls_NS[smoothing_level+1])+np.sum(np.sqrt(C_ls[smoothing_level+2:L+1]*V_ls_mlmc[smoothing_level+2:L+1]))) * 2 * epsilon ** (-2) * np.sqrt(V_ls_mlmc[l] / C_ls[l])) for l in range(smoothing_level+2, L+1)])))

        N_ls = N_ls.astype(int)

        print(f'New number of samples is {N_ls}.\n')

        ##### STEP 4 - Evaluate extra samples at each level
        # smoothed levels
        if (L <= smoothing_level+1):
            # in this case, we smooth all the levels, and only the last level has to not be smoothed
            for l in range(L):
                Y_hat_S_l = Y_hats_S[l]
                # Y_hat_NS_l = Y_ls_NS[l]
                N_old = len(Y_hat_S_l)

                # if we need more samples
                if(N_ls[l] > N_old):
                    Y_hat_S_l = np.resize(Y_hat_S_l, N_ls[l])
                    # Y_hat_unsmoothed_l = np.resize(Y_hat_unsmoothed_l, N_ls[l])
                    N_new_samples = N_ls[l] - N_old

                    print(f'New samples needed on level {l} is {N_new_samples}.')
                    print(f'Time estimated is {N_new_samples*time_per_sim[l] + (N_new_samples*time_per_sim[l-1] if (l!=0) else 0)} sec.')

                    t0 = time.time()

                    # compute extra sampled needed
                    _, _, Y_hat_S_extra = estimate_samples_LD.Y_l_smoothed_est_fun(N_new_samples, cov_fun, cov_fun_per, x, y, a, l, m_0, pol_degree, sigma, rho, nu, p)

                    t1 = time.time()
                    print(f'Time taken is {t1-t0} sec.')
                    
                    # save extra samples to re-use
                    Y_hat_S_l[N_old:] = Y_hat_S_extra
                    # Y_hat_unsmoothed_l[N_old:] = Y_hat_unsmoothed_extra

                    # Compute new expectation and variance and save in array
                    exp_S_l = np.average(Y_hat_S_l)
                    exp_ls_S[l] = exp_S_l

                    var_S_l = 1/(N_ls[l]-1) * np.sum((Y_hat_S_l - exp_ls_S[l]) ** 2)
                    V_ls_S[l] = var_S_l
                    Y_hats_S[l] = Y_hat_S_l
                    # Y_hats_unsmoothed[l] = Y_hat_unsmoothed_l

                    print(f"New expectation estimate is {np.sum(exp_ls_S[:L]) + exp_ls_NS[L]}. \n")

            # not smoothed level levels
            Y_hat_NS_l = Y_hats_NS[L]
            # Y_hat_NS_l = Y_ls_NS[l]
            N_old = len(Y_hat_NS_l)

            # if we need more samples
            if(N_ls[L] > N_old):
                Y_hat_NS_l = np.resize(Y_hat_NS_l, N_ls[L])
                # Y_hat_unsmoothed_l = np.resize(Y_hat_unsmoothed_l, N_ls[l])
                N_new_samples = N_ls[L] - N_old

                print(f'New samples needed on level {L} is {N_new_samples}.')
                print(f'Time estimated is {N_new_samples*time_per_sim[L] + (N_new_samples*time_per_sim[L-1] if (L!=0) else 0)} sec.')

                t0 = time.time()

                # compute extra sampled needed
                _, _, Y_hat_NS_extra = estimate_samples_LD.Y_l_not_smoothed_est_fun(N_new_samples, cov_fun, cov_fun_per, x, y, a, L, m_0, pol_degree, sigma, rho, nu, p)

                t1 = time.time()
                print(f'Time taken is {t1-t0} sec.')
                    
                # save extra samples to re-use
                Y_hat_NS_l[N_old:] = Y_hat_NS_extra
                # Y_hat_unsmoothed_l[N_old:] = Y_hat_unsmoothed_extra

                # Compute new expectation and variance and save in array
                exp_NS_l = np.average(Y_hat_NS_l)
                exp_ls_NS[L] = exp_NS_l

                var_NS_l = 1/(N_ls[L]-1) * np.sum((Y_hat_NS_l - exp_ls_NS[L]) ** 2)
                V_ls_NS[L] = var_NS_l
                Y_hats_NS[L] = Y_hat_NS_l
                # Y_hats_unsmoothed[l] = Y_hat_unsmoothed_l

                print(f"New expectation estimate is {np.sum(exp_ls_S[:L]) + exp_ls_NS[L]}. \n")
        
        else:
            # in this case, we smooth all the levels for which l <= ls, we do not smooth the ls+1 level (but ls is smoothed, so we need to take this into account), and use not smoothed differences for all the other levels
            for l in range(smoothing_level+1):
                # smoothed levels
                Y_hat_S_l = Y_hats_S[l]
                # Y_hat_NS_l = Y_ls_NS[l]
                N_old = len(Y_hat_S_l)
                # if we need more samples
                if(N_ls[l] > N_old):
                    Y_hat_S_l = np.resize(Y_hat_S_l, N_ls[l])
                    # Y_hat_unsmoothed_l = np.resize(Y_hat_unsmoothed_l, N_ls[l])
                    N_new_samples = N_ls[l] - N_old

                    print(f'New samples needed on level {l} is {N_new_samples}.')
                    print(f'Time estimated is {N_new_samples*time_per_sim[l] + (N_new_samples*time_per_sim[l-1] if (l!=0) else 0)} sec.')

                    t0 = time.time()

                    # compute extra sampled needed
                    _, _, Y_hat_S_extra = estimate_samples_LD.Y_l_smoothed_est_fun(N_new_samples, cov_fun, cov_fun_per, x, y, a, l, m_0, pol_degree, sigma, rho, nu, p)

                    t1 = time.time()
                    print(f'Time taken is {t1-t0} sec.')
                    
                    # save extra samples to re-use
                    Y_hat_S_l[N_old:] = Y_hat_S_extra
                    # Y_hat_unsmoothed_l[N_old:] = Y_hat_unsmoothed_extra

                    # Compute new expectation and variance and save in array
                    exp_S_l = np.average(Y_hat_S_l)
                    exp_ls_S[l] = exp_S_l

                    var_S_l = 1/(N_ls[l]-1) * np.sum((Y_hat_S_l - exp_ls_S[l]) ** 2)
                    V_ls_S[l] = var_S_l
                    Y_hats_S[l] = Y_hat_S_l
                    # Y_hats_unsmoothed[l] = Y_hat_unsmoothed_l

                    print(f"New expectation estimate is {np.sum(exp_ls_S[:L]) + exp_ls_NS[L]}. \n")
                
            # mixed between smoothed and not smoothed
            Y_hat_NS_l = Y_hats_NS[smoothing_level+1]
            N_old = len(Y_hat_NS_l)

            # if we need more samples
            if(N_ls[smoothing_level+1] > N_old):
                Y_hat_NS_l = np.resize(Y_hat_NS_l, N_ls[smoothing_level+1])
                # Y_hat_unsmoothed_l = np.resize(Y_hat_unsmoothed_l, N_ls[l])
                N_new_samples = N_ls[smoothing_level+1] - N_old

                print(f'New samples needed on level {smoothing_level+1} is {N_new_samples}.')
                print(f'Time estimated is {N_new_samples*time_per_sim[smoothing_level+1] + (N_new_samples*time_per_sim[smoothing_level] if (L!=0) else 0)} sec.')

                t0 = time.time()

                # compute extra sampled needed
                _, _, Y_hat_NS_extra = estimate_samples_LD.Y_l_not_smoothed_est_fun(N_new_samples, cov_fun, cov_fun_per, x, y, a, smoothing_level+1, m_0, pol_degree, sigma, rho, nu, p)

                t1 = time.time()
                print(f'Time taken is {t1-t0} sec.')
                    
                # save extra samples to re-use
                Y_hat_NS_l[N_old:] = Y_hat_NS_extra
                # Y_hat_unsmoothed_l[N_old:] = Y_hat_unsmoothed_extra

                # Compute new expectation and variance and save in array
                exp_NS_l = np.average(Y_hat_NS_l)
                exp_ls_NS[smoothing_level+1] = exp_NS_l

                var_NS_l = 1/(N_ls[smoothing_level+1]-1) * np.sum((Y_hat_NS_l - exp_ls_NS[smoothing_level+1]) ** 2)
                V_ls_NS[smoothing_level+1] = var_NS_l
                Y_hats_NS[smoothing_level+1] = Y_hat_NS_l
                # Y_hats_unsmoothed[l] = Y_hat_unsmoothed_l

                print(f"New expectation estimate is {np.sum(exp_ls_S[:smoothing_level+1]) + exp_ls_NS[smoothing_level+1]}. \n")
            
            for l in range(smoothing_level+2, L+1):
                # not smoothed levels
                Y_hat_mlmc_l = Y_hats_mlmc[l]
                # Y_hat_NS_l = Y_ls_NS[l]
                N_old = len(Y_hat_mlmc_l)
                # if we need more samples
                if(N_ls[l] > N_old):
                    Y_hat_mlmc_l = np.resize(Y_hat_mlmc_l, N_ls[l])
                    # Y_hat_unsmoothed_l = np.resize(Y_hat_unsmoothed_l, N_ls[l])
                    N_new_samples = N_ls[l] - N_old

                    print(f'New samples needed on level {l} is {N_new_samples}.')
                    print(f'Time estimated is {N_new_samples*time_per_sim[l] + (N_new_samples*time_per_sim[l-1] if (l!=0) else 0)} sec.')

                    t0 = time.time()

                    # compute extra sampled needed
                    _, _, Y_hat_mlmc_extra = estimate_samples.Y_l_est_fun(N_new_samples, x, y, a, l, m_0, cov_fun, cov_fun_per, pol_degree, sigma, rho, nu, p)

                    t1 = time.time()
                    print(f'Time taken is {t1-t0} sec.')
                    
                    # save extra samples to re-use
                    Y_hat_mlmc_l[N_old:] = Y_hat_mlmc_extra
                    # Y_hat_unsmoothed_l[N_old:] = Y_hat_unsmoothed_extra

                    # Compute new expectation and variance and save in array
                    exp_mlmc_l = np.average(Y_hat_mlmc_l)
                    exp_ls_mlmc[l] = exp_mlmc_l

                    var_mlmc_l = 1/(N_ls[l]-1) * np.sum((Y_hat_mlmc_l - exp_ls_mlmc[l]) ** 2)
                    V_ls_mlmc[l] = var_mlmc_l
                    Y_hats_mlmc[l] = Y_hat_mlmc_l
                    # Y_hats_unsmoothed[l] = Y_hat_unsmoothed_l

                    print(f"New expectation estimate is {np.sum(exp_ls_S[:smoothing_level+1]) + exp_ls_NS[smoothing_level] + exp_ls_mlmc[smoothing_level+2:L+1]} +. \n")


        # ### STEP 5 - Test for convergence if L >= 1

        # # Compute sample variance across all levels
        if (L <= smoothing_level+1):
            sample_var = np.sum(1 / N_ls[:L] * V_ls_S[:L]) + 1/N_ls[L] * V_ls_NS[L]
        else:
            sample_var = np.sum(1 / N_ls[:smoothing_level+1] * V_ls_S[:smoothing_level+1]) + 1 / N_ls[smoothing_level+1] * V_ls_NS[smoothing_level+1] + np.sum(1 / N_ls[smoothing_level+2:L+1] * V_ls_mlmc[smoothing_level+2:L+1])

        print(f'The sample variance is {sample_var} < {epsilon**2/2}.')

        # only test for converges if we have at least 2 levels (0 and 1)
        if (L >= 1):
            if (L <= smoothing_level+1):
                disc_error = (exp_ls_NS[L] / (1-4**alpha))**2
                # disc_error = (exp_ls_NS[L] / (1-C_tilde_alpha/C_alpha*4 **alpha))**2
            else:
                disc_error = (exp_ls_mlmc[L] / (1-4**alpha))**2
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
    if (L <= smoothing_level+1):
        exp_est = np.sum(exp_ls_S[:L]) + exp_ls_NS[L]
    else:
        exp_est = np.sum(exp_ls_S[:smoothing_level+1]) + exp_ls_NS[smoothing_level+1] + exp_ls_mlmc[smoothing_level+2:L+1]
    # compute total time taken
    total_time_t1 = time.time()
    total_time = total_time_t1-total_time_t0

    return Y_hat_S, Y_hat_NS, Y_hat_mlmc, exp_ls_S, exp_ls_NS, exp_ls_mlmc, V_ls_S, V_ls_NS, V_ls_mlmc, rmse, total_time, exp_est, N_ls

def MLMC_simulation(Y_ls_S, Y_ls_NS, Y_hat_S, Y_var_S, Y_hat_NS, Y_var_NS, x, y, a, epsilon, alpha, C_alpha, C_tilde_alpha, gamma, C_gamma, time_per_sim, cov_fun, cov_fun_per, m_0=4, pol_degree=1, sigma=1, rho=0.3, nu=0.5, p=1):
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
    C_ls = np.array([(np.exp(C_gamma))*(Ms[l]**gamma) + (np.exp(C_gamma))*(Ms[l-1]**gamma) for l in range(1,len(Ms))])
    C_ls = np.insert(C_ls, 0, (np.exp(C_gamma))*(Ms[0]**gamma))
    # arrays for saving quantities of interest
    V_ls_NS = Y_var_NS
    V_ls_S = Y_var_S
    exp_ls_NS = Y_hat_NS
    exp_ls_S = Y_hat_S
    Y_hats_NS = Y_ls_NS
    Y_hats_S = Y_ls_S

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

        print(f"Expectation estimate is {np.sum(exp_ls_S[:L])+exp_ls_NS[L]}.")
        
        ##### STEP 3 - Calculate optimal number of samples
        N_ls = np.array([math.ceil((np.sum(np.sqrt(C_ls[:L]*V_ls_S[:L]))+np.sqrt(C_ls[L]*V_ls_NS[L])) * 2 * epsilon ** (-2) * np.sqrt(V_ls_S[l] / C_ls[l])) for l in range(L)])

        N_ls = np.append(N_ls, math.ceil((np.sum(np.sqrt(C_ls[:L]*V_ls_S[:L]))+np.sqrt(C_ls[L]*V_ls_NS[L]))* 2 * epsilon ** (-2) * np.sqrt(V_ls_NS[L] / C_ls[L])))

        N_ls = N_ls.astype(int)

        print(f'New number of samples is {N_ls}.\n')

        ##### STEP 4 - Evaluate extra samples at each level
        # smoothed levels
        for l in range(L):
            Y_hat_S_l = Y_hats_S[l]
            # Y_hat_NS_l = Y_ls_NS[l]
            N_old = len(Y_hat_S_l)

            # if we need more samples
            if(N_ls[l] > N_old):
                Y_hat_S_l = np.resize(Y_hat_S_l, N_ls[l])
                # Y_hat_unsmoothed_l = np.resize(Y_hat_unsmoothed_l, N_ls[l])
                N_new_samples = N_ls[l] - N_old

                print(f'New samples needed on level {l} is {N_new_samples}.')
                print(f'Time estimated is {N_new_samples*time_per_sim[l] + (N_new_samples*time_per_sim[l-1] if (l!=0) else 0)} sec.')

                t0 = time.time()

                # compute extra sampled needed
                _, _, Y_hat_S_extra = estimate_samples_LD.Y_l_smoothed_est_fun(N_new_samples, cov_fun, cov_fun_per, x, y, a, l, m_0, pol_degree, sigma, rho, nu, p)

                t1 = time.time()
                print(f'Time taken is {t1-t0} sec.')
                    
                # save extra samples to re-use
                Y_hat_S_l[N_old:] = Y_hat_S_extra
                # Y_hat_unsmoothed_l[N_old:] = Y_hat_unsmoothed_extra

                # Compute new expectation and variance and save in array
                exp_S_l = np.average(Y_hat_S_l)
                exp_ls_S[l] = exp_S_l

                var_S_l = 1/(N_ls[l]-1) * np.sum((Y_hat_S_l - exp_ls_S[l]) ** 2)
                V_ls_S[l] = var_S_l
                Y_hats_S[l] = Y_hat_S_l
                # Y_hats_unsmoothed[l] = Y_hat_unsmoothed_l

                print(f"New expectation estimate is {np.sum(exp_ls_S[:L]) + exp_ls_NS[L]}. \n")

        # not smoothed level levels
        Y_hat_NS_l = Y_hats_NS[L]
        # Y_hat_NS_l = Y_ls_NS[l]
        N_old = len(Y_hat_NS_l)

        # if we need more samples
        if(N_ls[L] > N_old):
            Y_hat_NS_l = np.resize(Y_hat_NS_l, N_ls[L])
            # Y_hat_unsmoothed_l = np.resize(Y_hat_unsmoothed_l, N_ls[l])
            N_new_samples = N_ls[L] - N_old

            print(f'New samples needed on level {L} is {N_new_samples}.')
            print(f'Time estimated is {N_new_samples*time_per_sim[L] + (N_new_samples*time_per_sim[L-1] if (L!=0) else 0)} sec.')

            t0 = time.time()

            # compute extra sampled needed
            _, _, Y_hat_NS_extra = estimate_samples_LD.Y_l_not_smoothed_est_fun(N_new_samples, cov_fun, cov_fun_per, x, y, a, L, m_0, pol_degree, sigma, rho, nu, p)

            t1 = time.time()
            print(f'Time taken is {t1-t0} sec.')
                    
            # save extra samples to re-use
            Y_hat_NS_l[N_old:] = Y_hat_NS_extra
            # Y_hat_unsmoothed_l[N_old:] = Y_hat_unsmoothed_extra

            # Compute new expectation and variance and save in array
            exp_NS_l = np.average(Y_hat_NS_l)
            exp_ls_NS[L] = exp_NS_l

            var_NS_l = 1/(N_ls[L]-1) * np.sum((Y_hat_NS_l - exp_ls_NS[L]) ** 2)
            V_ls_NS[L] = var_NS_l
            Y_hats_NS[L] = Y_hat_NS_l
            # Y_hats_unsmoothed[l] = Y_hat_unsmoothed_l

            print(f"New expectation estimate is {np.sum(exp_ls_S[:L]) + exp_ls_NS[L]}. \n")

        # ### STEP 5 - Test for convergence if L >= 1

        # # Compute sample variance across all levels
        sample_var = np.sum(1 / N_ls[:L] * V_ls_S[:L]) + 1/N_ls[L] * V_ls_NS[L]
        print(f'The sample variance is {sample_var} < {epsilon**2/2}.')

        # only test for converges if we have at least 2 levels (0 and 1)
        if (L >= 1):
            disc_error = (exp_ls_NS[L] / (1-4**alpha))**2
            # disc_error = (exp_ls_NS[L] / (1-C_tilde_alpha/C_alpha*4 **alpha))**2
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
    exp_est = np.sum(exp_ls_S[:L]) + exp_ls_NS[L]
    # compute total time taken
    total_time_t1 = time.time()
    total_time = total_time_t1-total_time_t0

    return Y_hat_S, Y_hat_NS, exp_ls_S, exp_ls_NS, V_ls_S, V_ls_NS, rmse, total_time, exp_est, N_ls

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
    # smoothness parameter
    nu = 1.5
    # mesh size for coarsest level
    m_0_val = 4
    # norm to be used in covariance function of random field
    p_val = 2

    Y_hat_vec_S = np.load('./data/Y_hat_vec_S_LD_003_15_norm.npy', allow_pickle=True)
    Y_hat_var_vec_S = np.load('./data/Y_hat_var_vec_S_LD_003_15_norm.npy', allow_pickle=True)
    Y_ls_vec_S = np.load('./data/Y_ls_vec_S_LD_003_15_norm.npy', allow_pickle=True)

    Y_hat_vec_NS = np.load('./data/Y_hat_vec_NS_LD_003_15_norm.npy', allow_pickle=True)
    alpha_val = 0.3203869583916329
    C_alpha = -4.125424298666542
    Y_hat_var_vec_NS = np.load('./data/Y_hat_var_vec_NS_LD_003_15_norm.npy', allow_pickle=True)
    beta_val = 1.325511867482945
    C_beta = -0.935727804503858
    Y_ls_vec_NS = np.load('./data/Y_ls_vec_NS_LD_003_15_norm.npy', allow_pickle=True)

    avg_times_vec = np.load('./data/avg_times_vec_LD_15.npy', allow_pickle=True)
    avg_times_vec = avg_times_vec[1:]
    gamma_val = 0.8700621125059221
    C_gamma = -7.510443908031861

    C_alpha_mlmc = -3.631204069258538


    # list of different accuracies
    epsilon = 0.001

    print(f'Starting simulation for epsilon = {epsilon}.')

    # MLMC simulation
    Y_ls_vec_S, Y_ls_vec_NS, Y_hat_vec_S, Y_hat_vec_NS, Y_hat_var_vec_S, Y_hat_var_vec_NS, rmse_val, total_time_val, exp_est_val, N_ls_val = MLMC_simulation(Y_ls_vec_S, Y_ls_vec_NS, Y_hat_vec_S, Y_hat_var_vec_S, Y_hat_vec_NS, Y_hat_var_vec_NS, x_val, y_val, a_val, epsilon, alpha_val, C_alpha_mlmc, C_alpha, gamma_val, C_gamma, avg_times_vec, cov_functions.Matern_cov, periodisation_smooth.periodic_cov_fun, m_0_val, pol_degree_val, sigma, rho, nu, p_val)

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