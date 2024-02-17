from fenics import *
from pyfftw import *
from numpy.random import default_rng
import logging
import math
import numpy as np
from scipy.optimize import fsolve
# from tqdm import tqdm

import cov_functions
import PDE_solver
import circ_embedding
import periodisation_smooth

# suppress information messages (printing takes more time)
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.WARNING)

# generator for random variables
rng = default_rng()

def k_solver(k_l, m, alpha, C_alpha, C_tilde_alpha, nu):
    '''
    Equation to solve to find number of eigenvalues to drop per level.

    :param k_l: number of eigenvalues to drop per level
    :param m: mesh size
    :param alpha: the value of alpha required for Richardson extrapolation.
    :param C_alpha: the constant in |E[Q_l - Q_{l-1}]| = C_\alpha h^\alpha. 
    :param C_tilde_alpha: the constant in |E[Q_l - \tilde{Q}_{l-1}]| = \tilde{C}_\alpha h^\alpha. 
    :param nu: smoothness of the covariance function.

    :return:
        f - LHS for equation f(x) = 0 to solve.
    '''
    s=(2*m)**2 # in 2D
    C = (s - k_l)**(-(1+nu)/2)
    return C*k_l - np.exp(C_alpha) / 2*np.exp(C_tilde_alpha)*m**alpha*np.sqrt(s)

def Y_l_smoothed_est_fun(N, x, y, a, l, alpha, C_alpha, C_tilde_alpha, m_0, cov_fun, cov_fun_per, sigma=1, rho=0.3, nu=0.5, p=1, pol_degree=1):
    '''
    Computes the MC estimate of the expectation E[Y_l] and the variance V[Y_l] for a given level.

    :param N: number of samples to be used.
    :param x: the x value at which to compute E[p(x,y)].
    :param y: the y value at which to compute E[p(x,y)].
    :param a: the RHS constant.
    :param l: the level for which to compute E and V.
    :param alpha: the value of alpha required for Richardson extrapolation.
    :param C_alpha: the constant in |E[Q_l - Q_{l-1}]| = C_\alpha h^\alpha. 
    :param C_tilde_alpha: the constant in |E[Q_l - \tilde{Q}_{l-1}]| = \tilde{C}_\alpha h^\alpha. 
    :param m_0: the mesh size on level zero.
    :param cov_fun: covariance function.
    :param cov_fun_per: periodisation function to be used.
    :param sigma (default 1): variance of the covariance function.
    :param rho (default 0.3): correlation length of the covariance function.
    :param nu: smoothness of the covariance function.
    :param p: norm of covariance function argument.
    :param pol_degree (defualt 1): the degree of the polynomials used in FEM approximation.

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
    
    d = 2*m_per

    egnv = np.sqrt(egnv)
    egnv_tilde = np.zeros(d*d)
    indices = np.argsort(egnv)[::-1]
    k_l = fsolve(k_solver, 0, args=(m_per, alpha, C_alpha, C_tilde_alpha, nu))
    keep = int(d*d-k_l)
    egnv_tilde[indices[:keep]] = egnv[indices[:keep]]
    V_2, f_2, bc_2 = PDE_solver.setup_PDE(m_l, pol_degree, a)

    # MC loop - solve PDE for each k and compute average
    # for i in tqdm(range(N)): # Monte Carlo loop with status bar
    for i in range(N):
        # generate N random variables from standard Gaussian distribution
        xi = rng.standard_normal(size=d*d)

        # Compute solution on the two different grids (except for level 0) for current k
        if (l != 0):
            w = xi * egnv_tilde
            w = interfaces.scipy_fft.fft2(w.reshape((d, d)), norm='ortho')
            w = np.real(w) + np.imag(w)

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

            w = xi * egnv
            w = interfaces.scipy_fft.fft2(w.reshape((d, d)), norm='ortho')
            w = np.real(w) + np.imag(w)

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

        w = xi * egnv_tilde
        w = interfaces.scipy_fft.fft2(w.reshape((d, d)), norm='ortho')
        w = np.real(w) + np.imag(w)

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
        if (l != 0):
            p_hats[i] += (norm(u_2, 'L2') - norm(u, 'L2'))
            # p_hats[i] += u_2(x,y) - u(x,y)
        else:
            p_hats[i] += norm(u_2, 'L2')
            # p_hats[i] += u_2(x,y)

    # compute MC expectation
    Y_hat = np.average(p_hats)
    # compute MC sample variance
    Y_hat_var = 1 / (N-1) * np.sum((p_hats - Y_hat) ** 2)

    return np.abs(Y_hat), Y_hat_var, p_hats

def Y_l_not_smoothed_est_fun(N, x, y, a, l, alpha, C_alpha, C_tilde_alpha, m_0, cov_fun, cov_fun_per, sigma=1, rho=0.3, nu=0.5, p=1, pol_degree=1):
    '''
    Computes the MC estimate of the expectation E[Y_l] and the variance V[Y_l] for a given level.

    :param N: number of samples to be used.
    :param x: the x value at which to compute E[p(x,y)].
    :param y: the y value at which to compute E[p(x,y)].
    :param a: the RHS constant.
    :param l: the level for which to compute E and V.
    :param alpha: the value of alpha required for Richardson extrapolation.
    :param C_alpha: the constant in |E[Q_l - Q_{l-1}]| = C_\alpha h^\alpha. 
    :param C_tilde_alpha: the constant in |E[Q_l - \tilde{Q}_{l-1}]| = \tilde{C}_\alpha h^\alpha. 
    :param m_0: the mesh size on level zero.
    :param cov_fun: covariance function.
    :param cov_fun_per: periodisation function to be used.
    :param sigma (default 1): variance of the covariance function.
    :param rho (default 0.3): correlation length of the covariance function.
    :param nu: smoothness of the covariance function.
    :param p: norm of covariance function argument.
    :param pol_degree (defualt 1): the degree of the polynomials used in FEM approximation.

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

    d = 2*m_per
    egnv = np.sqrt(egnv)
    egnv_tilde = np.zeros(d*d)
    indices = np.argsort(egnv)[::-1]
    k_l = fsolve(k_solver, 0, args=(m_per, alpha, C_alpha, C_tilde_alpha, nu))
    keep = int(d*d-k_l)
    egnv_tilde[indices[:keep]] = egnv[indices[:keep]]
    V_2, f_2, bc_2 = PDE_solver.setup_PDE(m_l, pol_degree, a)

    # MC loop - solve PDE for each k and compute average
    # for i in tqdm(range(N)): # Monte Carlo loop with status bar
    for i in range(N):
        # generate N random variables from standard Gaussian distribution
        xi = rng.standard_normal(size=d*d)

        # Compute solution on the two different grids (except for level 0) for current k
        if (l != 0):
            w = xi * egnv_tilde
            w = interfaces.scipy_fft.fft2(w.reshape((d, d)), norm='ortho')
            w = np.real(w) + np.imag(w)

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

            w = xi * egnv
            w = interfaces.scipy_fft.fft2(w.reshape((d, d)), norm='ortho')
            w = np.real(w) + np.imag(w)

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

        w = xi * egnv
        w = interfaces.scipy_fft.fft2(w.reshape((d, d)), norm='ortho')
        w = np.real(w) + np.imag(w)

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
        if (l != 0):
            p_hats[i] += (norm(u_2, 'L2') - norm(u, 'L2'))
            # p_hats[i] += u_2(x,y) - u(x,y)
        else:
            p_hats[i] += norm(u_2, 'L2')
            # p_hats[i] += u_2(x,y)

    # compute MC expectation
    Y_hat = np.average(p_hats)
    # compute MC sample variance
    Y_hat_var = 1 / (N-1) * np.sum((p_hats - Y_hat) ** 2)

    return np.abs(Y_hat), Y_hat_var, p_hats

def Y_l_est_fun(N, cov_fun, cov_fun_per, x, y, a, l, m_0, pol_degree=1, sigma=1, rho=0.3, nu=0.5, p=1):
    '''
    Computes the MC estimate of the expectation E[Y_l] and the variance V[Y_l] for a given level.

    :param N: number of samples to be used.
    :param x: the x value at which to compute E[p(x,y)].
    :param y: the y value at which to compute E[p(x,y)].
    :param a: the RHS constant.
    :param l: the level for which to compute E and V.
    :param m_0: the mesh size on level zero.
    :param pol_degree (default 1): the degree of the polynomials used in FEM approximation.
    :param rho (default 0.3): correlation length of the covariance function.
    :param sigma (default 1): variance of the covariance function.

    :return:
        Y_hat - the expectation E[Y_l].
        Y_hat_var - the variance V[Y_l].
        p_hats - the samples Y_l used to compute approximations.
    '''
    
    m_l = 2**l*m_0
    print(f'm_l = {m_l}')
    # variable for saving the approximation on current grid
    p_hats = np.zeros(N)
    q_hats = np.zeros(N)
    p_smoothed = np.zeros(N)
    q_smoothed = np.zeros(N)

    # FEM setup for the 2 levels
    # treat level 0 differently as we only need one approximation
    if (l != 0):
        V, f, bc = PDE_solver.setup_PDE(m_l//2, pol_degree, a)

    egnv, m_per = circ_embedding.find_cov_eigenvalues(m_l, m_l, 2, cov_fun, cov_fun_per, sigma, rho, nu, p)
    egnv = np.sqrt(egnv)

    d = 2*m_per
    # k_ls = np.array([1, 2, 3, 8, 18, 42, 100, 241]) # rho = 0.1, nu = 1.5 point evaluation
    # k_ls = np.array([1, 2, 3, 5, 11, 24, 50, 107]) # rho = 0.1, nu = 1.5 L2 norm
    # k_ls = [48, 130, 483, 3000, 11429, 51437, 223212] # rho = 0.03, nu = 1.5 point evaluation
    k_ls = [48, 128, 600, 800, 8192, 45536, 205321] # rho = 0.03, nu = 1.5 L2 norm
    egnv_tilde = np.zeros(d*d)
    indices = np.argsort(egnv)[::-1]
    keep = int(d*d-k_ls[l])
    egnv_tilde[indices[:keep]] = egnv[indices[:keep]]
    V_2, f_2, bc_2 = PDE_solver.setup_PDE(m_l, pol_degree, a)
    
    smoothing_error = np.zeros(N)

    # MC loop - solve PDE for each k and compute average
    for i in tqdm(range(N)):
        # generate N random variables from standard Gaussian distribution
        xi = rng.standard_normal(size=d*d)

        # Compute solution on the two different grids (except for level 0) for current k
        if (l != 0):
            w = xi * egnv_tilde
            w = interfaces.scipy_fft.fft2(w.reshape((d, d)), norm='ortho')
            w = np.real(w) + np.imag(w)

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
        
        w = xi * egnv
        w = interfaces.scipy_fft.fft2(w.reshape((d, d)), norm='ortho')
        w = np.real(w) + np.imag(w)

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
        
        w = xi * egnv_tilde
        w = interfaces.scipy_fft.fft2(w.reshape((d, d)), norm='ortho')
        w = np.real(w) + np.imag(w)

        Z3 = w[:m_l+1, :m_l+1].reshape((m_l+1)*(m_l+1))

        smoothing_error[i] = np.linalg.norm(Z2-Z3, np.inf)

        # Compute solution for finer mesh (4*M)
        k_3 = PDE_solver.k_RF(Z3, m_l)
        u_3 = TrialFunction(V_2)
        v_3 = TestFunction(V_2)
        # F_2 = k_2 * dot(grad(u_2), grad(v_2)) * dx - f_2 * v_2 * dx
        a_3 = k_3 * dot(grad(u_3), grad(v_3)) * dx
        L_3 = f_2 * v_3 * dx
        u_3 = Function(V_2)
        # A_2, b_2 = assemble_system(a_2, L_2, bc_2)
        solve(a_3 == L_3, u_3, bc_2)

        # save current sample
        if (l != 0):
            p_hats[i] += (norm(u_2, 'L2') - norm(u, 'L2'))
            # p_hats[i] += u_2(x,y) - u(x,y)
            q_hats[i] += norm(u_2, 'L2')
            q_smoothed[i] +=  norm(u_3, 'L2')
            # q_smoothed[i] += u_3(x,y)
            p_smoothed[i] += (norm(u_3, 'L2') - norm(u, 'L2'))
            # p_smoothed[i] += u_3(x,y) - u(x,y)
        else:
            p_hats[i] += norm(u_2, 'L2')
            # p_hats[i] += u_2(x,y)
            p_smoothed[i] += norm(u_3, 'L2')
            # p_smoothed[i] += u_3(x,y)

    print(f'smoothing error = {np.average(smoothing_error)} \n')

    # compute MC expectation
    Y_hat = np.average(p_hats)
    # compute MC sample variance
    Y_hat_var = 1 / (N-1) * np.sum((p_hats - Y_hat) ** 2)

    Q_hat = np.average(q_hats)
    print(f'Q_hat = {Q_hat}')
    Q_hat_var = 1 / (N-1) * np.sum((q_hats - Q_hat) ** 2)
    print(f'Q_hat_var = {Q_hat_var} \n')

    Q_hat_smoothed = np.average(q_smoothed)
    print(f'Q_hat_smoothed = {Q_hat_smoothed}')
    Q_hat_var_smoothed = 1 / (N-1) * np.sum((q_smoothed - Q_hat_smoothed) ** 2)
    print(f'Q_hat_var_smoothed = {Q_hat_var_smoothed} \n')

    Y_hat_smoothed = np.average(p_smoothed)
    print(f'Y_hat_smoothed = {Y_hat_smoothed}')
    Y_hat_var_smoothed = 1 / (N-1) * np.sum((p_smoothed - Y_hat_smoothed) ** 2)
    print(f'Y_hat_var_smoothed = {Y_hat_var_smoothed}\n')

    return np.abs(Y_hat), Y_hat_var, p_hats, p_smoothed

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

    l = 6
    N = 150

    print(f'This is Level-Dependent MLMC with rho={rho} and nu={nu} and {N} samples on level {l}, iteration 01')

    Y_hat, Y_hat_var, Y_l_NS, Y_l_S = Y_l_est_fun(N, cov_functions.Matern_cov, periodisation_smooth.periodic_cov_fun, x_val, y_val, a_val, l, m_0_val, pol_degree_val, sigma, rho, nu, p_val)

    print(f'Y_hat_lev_{l} = {Y_hat}')
    print(f'Y_hat_var_lev_{l} = {Y_hat_var}')

    np.save(f'./data/Y_l_LD_NS_{l}_01.npy', Y_l_NS, allow_pickle=True)
    np.save(f'./data/Y_l_LD_S_{l}_01.npy', Y_l_S, allow_pickle=True)

if __name__ == "__main__":
    main()
