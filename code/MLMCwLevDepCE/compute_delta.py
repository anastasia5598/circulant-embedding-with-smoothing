#!/work/sc073/sc073/s2079009/miniconda3/bin/python

from fenics import *
from pyfftw import *
from numpy.random import default_rng
import numpy as np
import time
import logging
import sys
sys.path.insert(0, '/home/s2079009/MAC-MIGS/PhD/PhD_code/')
from utils import cov_functions
from utils import PDE_solver
from utils import circ_embedding

# suppress information messages (printing takes more time)
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.WARNING)

# generator for random variables
rng = default_rng()

def compute_delta(N, J1, J2, cov_fun, x, y, a_val, m_0, pol_degree=1, sigma=1, rho=0.3, nu=0.5, p=1):
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
    Y_hats = np.zeros(5)
    Y_hat_vars = np.zeros(5)
    m_ls = np.zeros(5)
    Y_ls = []

    # use 7 levels - from 0 to 6
    for l in range(1,6):
        m_l = 64
        # variable for saving the approximation on current grid
        p_hats = np.zeros(N)

        # FEM setup for the 2 levels
        V, f, bc = PDE_solver.setup_PDE(m_l//2, pol_degree, a_val)
        V_2, f_2, bc_2 = PDE_solver.setup_PDE(m_l, pol_degree, a_val)

        egnv, d, _ = circ_embedding.find_cov_eigenvalues(m_l, m_l, J1, J2, cov_fun, sigma, rho, nu, p)
        egnv = np.sqrt(np.real(egnv))
        egnv_tilde = np.zeros(d*d)
        indices = np.argsort(egnv)[::-1]
        egnv_tilde[indices[:d*d//8]] = egnv[indices[:d*d//8]]

        # MC loop - solve PDE for each k and compute average
        for i in range(N):
            # generate N random variables from standard Gaussian distribution
            xi = rng.standard_normal(size=d*d)

            w = xi * egnv_tilde
            w = interfaces.scipy_fft.fft2(w.reshape((d, d)), norm='ortho')
            w = np.real(w) + np.imag(w)

            # Compute solution on the two different grids (except for level 0) for current k
            Z1 = w[:m_l+1:2, :m_l+1:2].reshape(((m_l//2)+1)*((m_l//2)+1))

            # Compute solution for initial mesh-size (M)
            k = PDE_solver.k_RF(Z1, m_l//2)
            u = TrialFunction(V)
            v = TestFunction(V)
            # F = k * dot(grad(u), grad(v)) * dx - f * v * dx
            a = k * dot(grad(u), grad(v)) * dx
            L = f * v * dx
            u = Function(V)
            # A, b = assemble_system(a, L, bc)
            solve(a == L, u, bc)

            Z2 = w[:m_l+1, :m_l+1].reshape((m_l+1)*(m_l+1))

            # Compute solution for finer mesh (4*M)
            k_2 = PDE_solver.k_RF(Z2, m_l)
            u_2 = TrialFunction(V_2)
            v_2 = TestFunction(V_2)
            # F_2 = k_2 * dot(grad(u_2), grad(v_2)) * dx - f_2 * v_2 * dx
            a_2 = k_2 * dot(grad(u_2), grad(v_2)) * dx
            L_2 = f_2 * v_2 * dx
            u_2 = Function(V_2)
            # A_2, b_2 = assemble_system(a_2, L_2, bc_2)
            solve(a_2 == L_2, u_2, bc_2)

            # save current sample
            p_hats[i] += (norm(u, 'L2') - norm(u_2, 'L2'))

        # compute MC expectation
        Y_hat = np.average(p_hats)
        # compute MC sample variance
        Y_hat_var = 1 / (N-1) * np.sum((p_hats - Y_hat) ** 2)

        # save current values
        m_ls[l-1] = 2**l
        Y_hats[l-1] = np.abs(Y_hat)   
        Y_hat_vars[l-1] = Y_hat_var
        Y_ls.append(p_hats)

    # compute alpha and C_alpha by finding best linear fit through Y_hats
    delta, C_delta = np.polyfit(x=np.log(m_ls), y=np.log(Y_hats), deg=1)

    return m_ls, -delta, C_delta, Y_hats

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

    m_vec, delta_val, C_delta, Y_hat_vec = compute_delta(N=2, J1=J1_val, J2=J2_val, cov_fun=cov_functions.expopnential_cov, x=x_val, y=y_val, a_val=a_val, m_0=m_0_val, pol_degree=pol_degree_val, sigma=sigma, rho=rho, nu=nu, p=p_val)


    print(f'm_vec={m_vec}')
    print(f'delta_val={delta_val}')
    print(f'C_delta={C_delta}')

    np.save(f'./data/Y_hat_vec.npy', Y_hat_vec, allow_pickle=True)

if __name__ == "__main__":
    main()