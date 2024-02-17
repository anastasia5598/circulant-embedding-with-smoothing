from fenics import *
from pyfftw import *
from numpy.random import default_rng
import numpy as np
import time
import math
import logging
# from tqdm import tqdm

import PDE_solver
import cov_functions
import circ_embedding
import periodisation_smooth
import estimate_samples

# suppress information messages (printing takes more time)
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.WARNING)

# generator for random variables
rng = default_rng()

def var_est_fun(N, x, y, a, cov_fun, cov_fun_per, sigma, rho, nu, p, mesh_size=16, pol_degree=1):
    '''
    Computes the sample variance V[Q] on a fine enough mesh and with a large enough number of samples, so that it can be considered almost exact.
    
    :param N: number of samples to be used.
    :param x: the x value at which to compute E[p(x,y)].
    :param y: the y value at which to compute E[p(x,y)].
    :param a: the RHS constant.
    :param cov_fun: covariance function.
    :param cov_fun_per: periodisation function to be used.
    :param sigma (default 1): variance of the covariance function.
    :param rho (default 0.3): correlation length of the covariance function.
    :param nu: smoothness of the covariance function.
    :param p: norm of covariance function argument.
    :param mesh_size (default 16): the stepsize to be used in approximation.
    :param pol_degree (defualt 1): the degree of the polynomials used in FEM approximation.

    :return:
        var - the sample variance.
    '''
    # numpy array for saving approximations
    p_sol = np.zeros(N)
    # setup FEM 
    V, f, bc = \
        PDE_solver.setup_PDE(m=mesh_size, pol_degree=pol_degree, a=a)
    egnv, m_per = \
        circ_embedding.find_cov_eigenvalues(mesh_size, mesh_size, 2, cov_fun, cov_fun_per, sigma, rho, nu, p)
    
    # MC loop - solve PDE for each k and average results
    # for i in tqdm(range(N)): # Monte Carlo loop with status bar
    for i in range(N):
        # generate N random variables from log-normal distribution
        xi = rng.standard_normal(size=4*m_per*m_per)
        w = xi * (np.sqrt(np.real(egnv)))
        w = interfaces.scipy_fft.fft2(w.reshape((2*m_per, 2*m_per)), norm='ortho')
        w = np.real(w) + np.imag(w)    
        Z = w[:mesh_size+1, :mesh_size+1].reshape((mesh_size+1)*(mesh_size+1))

        # Compute solution with current k and mesh size
        k = PDE_solver.k_RF(Z, mesh_size)
        u = Function(V)
        v = TestFunction(V)
        F = k * dot(grad(u), grad(v)) * dx - f * v * dx
        solve(F == 0, u, bc)

        # save value p(x,y)
        p_sol[i] += u(x,y)
        # p_sol[i] += norm(u, 'L2')

    # MC estimate 
    p_hat = np.average(p_sol)
    # sample variance
    var = 1 / (N-1) * np.sum((p_sol - p_hat) ** 2)

    return var

def compute_alpha(x, y, a, m_0, cov_fun, cov_fun_per, sigma=1, rho=0.3, nu=0.5, p=1, pol_degree=1):
    '''
    Computes alpha in E[Q_M - Q] = C * M ^ (-alpha). Uses Y_l_est_fun.

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
    '''

    # empty lists for storing the approximations and grid size.
    Y_hats = np.zeros(7)
    Ms = np.zeros(7)

    # number of samples needed to compute initial estimate of sample variance on each level
    Ns = [10000, 5000, 2500, 1250, 625, 315, 160]

    # use 7 levels - from 0 to 6
    for l in range(7):
        # for each level, compute expectation and variance
        Y_hat, _ , _ = \
            estimate_samples.Y_l_est_fun(Ns[l], x, y, a, l, m_0, cov_fun, cov_fun_per, sigma, rho, nu, p, pol_degree)

        # save current values
        Ms[l] = (m_0 * 2**l) ** 2
        Y_hats[l] = np.abs(Y_hat)   

    # compute alpha and C_alpha by finding best linear fit through Y_hats
    alpha, C_alpha = np.polyfit(x=np.log(Ms[1:]), y=np.log(Y_hats[1:]), deg=1)

    return -alpha, C_alpha

def MC_simulation(N, x, y, a, alpha, var, cov_fun, cov_fun_per, mesh_size=8, pol_degree=1, sigma=1, rho=0.3, nu=0.5, p=1):
    '''
    Performs a standard Monte Carlo simulation for computing E[p(x,y)].

    :param N: the number of samples to use per simulation.
    :param x: the x point at which to compute E[p(x,y)].
    :param y: the y point at which to compute E[p(x,y)].
    :param a: the RHS constant of the PDE.
    :param alpha: constant for computing E[Q_M - Q].
    :param var: the sample variance used in computing RMSE.
    :param cov_fun: covariance function.
    :param cov_fun_per: periodisation function to be used.
    :param mesh_size (default 8): the step size to be used for computing FEM approximation.
    :param pol_degree (default 1): degree of polynomial to be used to computed FEM approximation.
    :param rho (default 0.3): correlation length of the covariance function.
    :param sigma (default 1): variance of the covariance function.
    :param nu (default 0.5): smoothness parameter of covariance function.

    :return: 
        rmse - root mean squared error between approximation and exact value.
        total_time - time taken to compute MC estimate.
    '''

    # values for computing approximation on 2 different grids
    p_hat = 0
    p_hat_2 = 0

    # time the duration of simulation
    total_time = 0

    t0 = time.time()
    egnv, m_per = \
        circ_embedding.find_cov_eigenvalues(mesh_size, mesh_size, 2, cov_fun, cov_fun_per, sigma, rho, nu, p)
    

    # FEM setup on two different grids
    V, f, bc = \
        PDE_solver.setup_PDE(m=mesh_size, pol_degree=pol_degree, a=a)

    V_2, f_2, bc_2 = \
        PDE_solver.setup_PDE(m=mesh_size//2, pol_degree=pol_degree, a=a)

    # Monte Carlo simulation for computing E[p(x,y)] 
    # for i in tqdm(range(N)): # use this for loop with status bar
    for i in range(N):

        xi = rng.standard_normal(size=4*m_per*m_per)
        w = xi * (np.sqrt(np.real(egnv)))
        w = interfaces.scipy_fft.fft2(w.reshape((2*m_per, 2*m_per)), norm='ortho')
        w = np.real(w) + np.imag(w)
        Z1 = w[:mesh_size+1, :mesh_size+1].reshape((mesh_size+1)*(mesh_size+1))

        # Compute solution on current grid with current k
        k = PDE_solver.k_RF(Z1, mesh_size)
        u = Function(V)
        v = TestFunction(V)
        F = k * dot(grad(u), grad(v)) * dx - f * v * dx
        solve(F == 0, u, bc)
        
        # quantity of interest
        p_hat += u(x, y)
        # p_hat += norm(u, 'L2')

        # Compute solution for finer mesh (4*M)
        Z2 = w[:mesh_size+1:2, :mesh_size+1:2].reshape(((mesh_size//2)+1)*((mesh_size//2)+1))

        # Compute solution for finer mesh (4*M)
        k_2 = PDE_solver.k_RF(Z2, mesh_size//2)
        u_2 = Function(V_2)
        v_2 = TestFunction(V_2)
        F_2 = k_2 * dot(grad(u_2), grad(v_2)) * dx - f_2 * v_2 * dx
        solve(F_2 == 0, u_2, bc_2)

        # quantity of interest
        p_hat_2 += u_2(x, y)
        # p_hat_2 += norm(u_2, 'L2')

    # Monte Carlo estimates of E[p(x, y)]
    exp_est = p_hat/N
    exp_est_2 = p_hat_2/N

    # extrapolation value of discretisation error E[Q_M-Q]
    exp_true = (exp_est_2-exp_est) / (1 - 4 ** (-alpha))
    # compute discretisation error
    disc_error = exp_true ** 2  
    
    # Root Mean Squared Error of E[p(x,y)]
    rmse = np.sqrt(1 / N * var + disc_error)

    t1 = time.time()
    total_time = t1-t0

    return rmse, total_time

# below is an example of running a Monte Carlo simulation for a given accuracy
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
#     # smoothness parameters of random field
#     nu = 1.5
#     # norm to be used in covariance function of random field
#     p_val = 2
#     # mesh size for coarsest level
#     m_0_val = 4

#     # variance is assumed to be constant, and so we can use a relatively small number of samples and a small mesh size to estimate it 
#     # var_val = var_est_fun(N=100, x=x_val, y=y_val, a=a_val, cov_fun=cov_functions.Matern_cov, cov_fun_per=periodisation_smooth.periodic_cov_fun, sigma=sigma, rho=rho, nu=nu, p=p_val, mesh_size=128, pol_degree=pol_degree_val)

#     # estimate alpha and C_alpha in the discretisation error assumption
#     alpha_val, C_alpha = compute_alpha(x=x_val, y=y_val, a=a_val, m_0=m_0_val, cov_fun=cov_functions.Matern_cov, cov_fun_per=periodisation_smooth.periodic_cov_fun, sigma=sigma, rho=rho, nu=nu, p=p_val, pol_degree=pol_degree_val)

#     # these are the pre-computed values which you can use
#     # var_val = 0.0018769639289535666
#     # alpha_val = 0.3570868564486739
#     # C_alpha = -3.053440188533286

#     #  accuracy
#     epsilon = 0.05

#     mesh_sizes = [2**n for n in range(4, 9)]
        
#     print(f'Starting simulation for epsilon = {epsilon}.')
#     # number of samples needed to reach epsilon accuracy
#     N_val = 3*math.ceil(2*epsilon ** (-2) * var_val)
#     # mesh size needed to reach epsilon accuracy
#     mesh_size_val = \
#         math.ceil((math.sqrt(2)*np.exp(C_alpha)/epsilon)**(1/(2*alpha_val)))
#     mesh_size_val = min(mesh_sizes, key=lambda x:abs(x-mesh_size_val))
#     print(f'N = {N_val} and mesh size = {mesh_size_val}.')

#     # MC simulation for current epsilon
#     rmse_val, time_val = \
#         MC_simulation(N_val, x_val, y_val, a_val, alpha_val, var_val, cov_functions.Matern_cov, periodisation_smooth.periodic_cov_fun, mesh_size_val, pol_degree_val, sigma, rho, nu, p_val)

#     print(f'rmse_val = {rmse_val}')
#     print(f'time_val = {time_val}')