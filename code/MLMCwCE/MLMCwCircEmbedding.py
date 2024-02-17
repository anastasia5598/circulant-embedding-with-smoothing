from fenics import *
from pyfftw import *
from numpy.random import default_rng
import numpy as np
import math
import time
import logging
# from tqdm import tqdm

import circ_embedding
import PDE_solver
import estimate_samples
import cov_functions
import periodisation_smooth

# suppress information messages (printing takes more time)
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.WARNING)

# generator for random variables
rng = default_rng()

def compute_alpha_beta(x, y, a, m_0, cov_fun, cov_fun_per, sigma=1, rho=0.3, nu=0.5, p=1, pol_degree=1):
    '''
    Computes alpha in E[Q_M - Q] = C * M ^ (-alpha) and beta in V[Q_M-Q] = C * M ^ (-beta) by computing approximations on different levels. Uses Y_l_est_fun.

    :param x: the x point at which to compute E[p(x,y)].
    :param y: the y point at which to compute E[p(x,y)].
    :param a: the RHS constant of the PDE.
    :param m_0: mesh size on level zero.
    :param cov_fun: covariance function.
    :param cov_fun_per: periodisation function to be used.
    :param sigma (default 1): variance of the covariance function.
    :param rho (default 0.3): correlation length of the covariance function.
    :param nu: smoothness of the covariance function.
    :param p: norm of covariance function argument.
    :param pol_degree (defualt 1): the degree of the polynomials used in FEM approximation.

    :return: 
        alpha - slope of Y_hats line on log-log scale. 
        C_alpha - constant in E[Q_M - Q] = C * M ^ (-alpha).
        beta - slope of Y_hat_vars line on log-log scale.
        C_beta - contant in V[Q_M-Q] = C * M ^ (-beta).
    '''

    # empty lists for storing the approximations and grid size.
    Y_hats = np.zeros(7)
    Y_hat_vars = np.zeros(7)
    Ms = np.zeros(7)
    Y_ls = []

    # number of samples needed to compute initial estimate of sample variance on each level
    Ns = [10000, 5000, 2500, 1250, 625, 315, 160]

    # use 7 levels - from 0 to 6
    for l in range(7):
        # for each level, compute expectation and variance
        Y_hat, Y_hat_var, Y_l = \
            estimate_samples.Y_l_est_fun(Ns[l], x, y, a, l, m_0, cov_fun, cov_fun_per, sigma, rho, nu, p, pol_degree)

        # save current values
        Ms[l] = (m_0 * 2**l) ** 2
        Y_hats[l] = np.abs(Y_hat)   
        Y_hat_vars[l] = Y_hat_var
        Y_ls.append(Y_l)

    # compute alpha and C_alpha by finding best linear fit through Y_hats
    alpha, C_alpha = np.polyfit(x=np.log(Ms[1:]), y=np.log(Y_hats[1:]), deg=1)
    # compute beta and C_beta by finding best linear fit through Y_hat_vars
    beta, C_beta = np.polyfit(x=np.log(Ms[1:]), y=np.log(Y_hat_vars[1:]), deg=1)

    return -alpha, C_alpha, -beta, C_beta

def compute_gamma(N, a, cov_fun, cov_fun_per, sigma=1, rho=0.3, nu=0.5, p=1, pol_degree=1):
    """
    Computes gamma in Cost = C * M ^ gamma by computing multiple approximations 
    on different grids and averaging over N runs.

    :param N: the number of times to run one Monte Carlo simulation.
    :param a: the RHS constant of the PDE.
    :param cov_fun: covariance function.
    :param cov_fun_per: periodisation function to be used.
    :param sigma (default 1): variance of the covariance function.
    :param rho (default 0.3): correlation length of the covariance function.
    :param nu: smoothness of the covariance function.
    :param p: norm of covariance function argument.
    :param pol_degree (defualt 1): the degree of the polynomials used in FEM approximation.

    :return: 
        gamma - the slope on the log-log scale of times.
        C_gamma -  the constant in Cost = C * M^gamma.
    """

    # empty lists for storing the times and grid size.
    Ms = np.zeros(7)
    avg_times = np.zeros(7)

    # use 7 different stepsizes - from 2^2 to 2^8
    for n in range(2, 9):
        times = np.zeros(N)

        # FEM setup for current mesh size
        m=2**n
        egnv, m_per = circ_embedding.find_cov_eigenvalues(m, m, 2, cov_fun, cov_fun_per, sigma, rho, nu, p)

        V, f, bc = PDE_solver.setup_PDE(m, pol_degree, a)

        # MC loop - sample k and solve PDE for current k
        # for i in tqdm(range(N)): # Monte Carlo loop with status bar
        for i in range(N):
            # generate N random variables from standard Gaussian distribution
            t0 = time.time()

            xi = rng.standard_normal(size=4*m_per*m_per)
            w = xi * (np.sqrt(np.real(egnv)))
            w = interfaces.scipy_fft.fft2(w.reshape((2*m_per, 2*m_per)), norm='ortho')
            w = np.real(w) + np.imag(w)    
            Z1 = w[:m+1, :m+1].reshape((m+1)*(m+1))

            # Compute solution for initial mesh-size (M)
            k = PDE_solver.k_RF(Z1, m)
            u = Function(V)
            v = TestFunction(V)
            F = k * dot(grad(u), grad(v)) * dx - f * v * dx
            solve(F == 0, u, bc) 

            t1 = time.time()
            times[i] = t1-t0

        # save current values
        Ms[n-2] = 4 ** n
        avg_times[n-2] = np.average(times)

    # compute gamma and C by finding best linear fit on log-log scale
    gamma, C_gamma = np.polyfit(x=np.log(Ms), y=np.log(avg_times), deg=1)

    return gamma, C_gamma


def MLMC_simulation(x, y, a, epsilon, alpha, gamma, C_gamma, m_0, cov_fun, cov_fun_per, sigma=1, rho=0.3, nu=0.5, p=1, pol_degree=1):
    '''
    Runs one simulation of MLMC for computing E[p(x,y)]. In particular, implements MLMC algorithm in Cliffe (2011).

    :param x: the x value at which to compute E[p(x,y)].
    :param y: the y value at which to compute E[p(x,y)].
    :param a: the RHS constant.
    :param epsilon: the required accuracy.
    :param alpha: the value of alpha required for Richardson extrapolation.
    :param gamma: the slope of the cost required per simulation.
    :param C_gamma: the constant in Cost = C * M ^ gamma.
    :param m_0: the mesh size on the coarsest level.
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
    C_ls = np.array([(np.exp(C_gamma))*(Ms[l]**gamma) + (np.exp(C_gamma))*(Ms[l-1]**gamma) for l in range(1,7)])
    C_ls = np.insert(C_ls, 0, (np.exp(C_gamma))*(Ms[0]**gamma))
    Ns = [5000, 4000, 3000, 1500, 750, 375, 187]
    # arrays for saving quantities of interest
    V_ls = []
    exp_ls = []
    Y_hats = []

    ##### STEP 1 - Start with L = 0
    L = 0

    total_time_t0 = time.time()

    while(True):
        ##### STEP 2 - Estimate variance using initial number of samples
        Y_hat, Y_hat_var, Y_l = \
            estimate_samples.Y_l_est_fun(Ns[L], x, y, a, L, m_0, cov_fun, cov_fun_per, sigma, rho, nu, p)

        # # save initial variance, expectation and samples computed
        V_ls.append(Y_hat_var)
        exp_ls.append(Y_hat)
        Y_hats.append(Y_l)

        ##### STEP 3 - Calculate optimal number of samples
        N_ls = np.array([ math.ceil(\
            np.sum( np.sqrt( C_ls[:L+1] * V_ls[:L+1] ) ) \
                * 2 * epsilon ** (-2) * np.sqrt( V_ls[l] / C_ls[l] ) ) \
                    for l in range(L+1)])

        ##### STEP 4 - Evaluate extra samples at each level
        for l in range(L+1):

            Y_hat_l = Y_hats[l]
            N_old = len(Y_hat_l)

            # if we need more samples
            if(N_ls[l] > N_old):
                Y_hat_l = np.resize(Y_hat_l, N_ls[l])
                N_new_samples = N_ls[l] - N_old

                # compute extra sampled needed
                _, _, Y_hat_extra = estimate_samples.Y_l_est_fun(N_new_samples, x, y, a, l, m_0, cov_fun, cov_fun_per, sigma, rho, nu, p, pol_degree)
                
                # save extra samples to re-use
                Y_hat_l[N_old:] = Y_hat_extra

                # Compute new expectation and variance and save in array
                exp_l = np.average(Y_hat_l)
                var_l = 1 / (N_ls[l]-1) * np.sum((Y_hat_l - exp_ls[l]) ** 2)

                V_ls[l] = var_l
                exp_ls[l] = exp_l
                Y_hats[l] = Y_hat_l

        ### STEP 5 - Test for convergence if L >= 1

        # Compute sample variance across all levels
        sample_var = np.sum(1 / N_ls * V_ls[:L+1])

        # only test for converges if we have at least 2 levels (0 and 1)
        if (L >= 1):
            # compute discretisation error
            disc_error = (exp_ls[L] / (1-4 ** alpha))**2

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

    # compute total time taken
    total_time_t1 = time.time()
    total_time = total_time_t1-total_time_t0

    return rmse, total_time

# below is an example of running a Multilevel Monte Carlo simulation for a given accuracy
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
#     # smoothness parameters
#     nu = 1.5
#     # mesh size for coarsest level
#     m_0_val = 16
#     # norm to be used in covariance function of random field
#     p_val = 2

#     alpha_val, C_alpha, beta_val, C_beta = compute_alpha_beta(x=x_val, y=y_val, a=a_val, m_0=m_0_val, cov_fun=cov_functions.Matern_cov, cov_fun_per=periodisation_smooth.periodic_cov_fun, sigma=sigma, rho=rho, nu=nu, p=p_val, pol_degree=pol_degree_val)
    
#     gamma_val, C_gamma = compute_gamma(100, a_val, cov_functions.Matern_cov, periodisation_smooth.periodic_cov_fun, sigma, rho, nu, p_val, pol_degree_val)

#     # these are the pre-computed values which you can use
#     # alpha_val = 0.3551417357831946
#     # C_alpha = -3.631204069258538
#     # beta_val = 1.5666865274225823
#     # C_beta = 1.470065536870814
#     # gamma_val = 0.9868719259553745
#     # C_gamma = -8.510608041617592

#     epsilon = 0.00075 # accuracy

#     # MLMC simulation
#     rmse_val, total_time_val = MLMC_simulation(x_val, y_val, a_val, epsilon, alpha_val, gamma_val, C_gamma, m_0_val, cov_functions.Matern_cov, periodisation_smooth.periodic_cov_fun, sigma, rho, nu, p_val, pol_degree_val)

#     print(f'Total time is {total_time_val}.')
#     print(f'Final RMSE is {rmse_val} < {epsilon}.')