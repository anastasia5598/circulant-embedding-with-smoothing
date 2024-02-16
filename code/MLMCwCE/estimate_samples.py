from fenics import *
from pyfftw import *
from numpy.random import default_rng
import numpy as np
import logging
# from tqdm import tqdm

import circ_embedding
import cov_functions
import PDE_solver
import periodisation_smooth

# suppress information messages (printing takes more time)
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.WARNING)

# generator for random variables
rng = default_rng()

def Y_l_est_fun(N, x, y, a, l, m_0, cov_fun, cov_fun_per, sigma=1, rho=0.3, nu=0.5, p=1, pol_degree=1):
    '''
    Computes the MC estimate of the expectation E[Y_l] and the variance V[Y_l] for a given level.

    :param N: number of samples to be used.
    :param x: the x value at which to compute E[p(x,y)].
    :param y: the y value at which to compute E[p(x,y)].
    :param a: the RHS constant.
    :param l: the level for which to compute E and V.
    :param m_0: the mesh size on level zero.
    :param cov_fun: covariance function.
    :param cov_fun_per: periodisation function to be used.
    :param sigma (default 1): variance of the covariance function.
    :param rho (default 0.3): correlation length of the covariance function.
    :param nu: smoothness of the covariance function.
    :param p: norm of covariance function argument.
    :param pol_degree (default 1): the degree of the polynomials used in FEM approximation.

    :return:
        Y_hat - the expectation E[Y_l].
        Y_hat_var - the variance V[Y_l].
        p_hats - the samples Y_l used to compute approximations.
    '''
    
    m_l = 2**l*m_0
    # variable for saving the approximation on current grid
    p_hats = np.zeros(N)

    # FEM setup for the 2 levels
    # treat level 0 differently as we only need one approximation
    if (l != 0):
        V, f, bc = PDE_solver.setup_PDE(m_l//2, pol_degree, a)

    egnv, m_per = circ_embedding.find_cov_eigenvalues(m_l, m_l, 2, cov_fun, cov_fun_per, sigma, rho, nu, p)
    egnv = np.sqrt(egnv)

    V_2, f_2, bc_2 = PDE_solver.setup_PDE(m_l, pol_degree, a)

    # MC loop - solve PDE for each k and compute average
    # for i in tqdm(range(N)): # Monte Carlo loop with status bar
    for i in range(N):
        # generate N random variables from standard Gaussian distribution
        xi = rng.standard_normal(size=4*m_per*m_per)
        w = xi * egnv
        w = interfaces.scipy_fft.fft2(w.reshape((2*m_per, 2*m_per)), norm='ortho')
        w = np.real(w) + np.imag(w)    

        # Compute solution on the two different grids (except for level 0) for current k
        if (l != 0):
            Z1 = w[:m_l+1:2, :m_l+1:2].reshape(((m_l//2)+1)*((m_l//2)+1))

            # Compute solution for initial mesh-size (M)
            k = PDE_solver.k_RF(Z1, m_l//2)
            u = Function(V)
            v = TestFunction(V)
            F = k * dot(grad(u), grad(v)) * dx - f * v * dx
            solve(F == 0, u, bc)

        Z2 = w[:m_l+1, :m_l+1].reshape((m_l+1)*(m_l+1))

        # Compute solution for finer mesh (4*M)
        k_2 = PDE_solver.k_RF(Z2, m_l)
        u_2 = Function(V_2)
        v_2 = TestFunction(V_2)
        F_2 = k_2 * dot(grad(u_2), grad(v_2)) * dx - f_2 * v_2 * dx
        solve(F_2 == 0, u_2, bc_2)

        # save current sample
        if (l != 0):
            p_hats[i] += (u_2(x, y) - u(x, y))
            # p_hats[i] += (norm(u_2, 'L2') - norm(u, 'L2'))
        else:
            p_hats[i] += u_2(x, y)
            # p_hats[i] += norm(u_2, 'L2')

    # compute MC expectation
    Y_hat = np.average(p_hats)

    # compute MC sample variance
    Y_hat_var = 1 / (N-1) * np.sum((p_hats - Y_hat) ** 2)

    return np.abs(Y_hat), Y_hat_var, p_hats