from fenics import *
from pyfftw import *
from numpy.random import default_rng
import numpy as np
import time
import logging
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/s2079009/MAC-MIGS/PhD/PhD_code/')
from utils import circ_embedding
from utils import cov_functions
from utils import PDE_solver
from utils import periodisation_smooth

# suppress information messages (printing takes more time)
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.WARNING)

# generator for random variables
rng = default_rng()

def compute_gamma(N, a, cov_fun, cov_fun_per, pol_degree=1, rho=0.3, sigma=1, nu=0.5, p=1):
    """
    Computes gamma in Cost = C * M ^ gamma by computing multiple approximations 
    on different grids and averaging over N runs.

    :param N: the number of times to run one Monte Carlo simulation.
    :param m_KL: number of modes to include in KL-expansion of random field.
    :param a: the RHS constant of the PDE.
    :param rho (default 0.3): correlation length of the covariance function.
    :param sigma (default 1): variance of the covariance function.

    :return: 
        Ms - different grid sizes used in computing approximations.
        avg_times - ndarray of time taken on average for one simulation.
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
        print(m)
        egnv, m_per = circ_embedding.find_cov_eigenvalues(m, m, 2, cov_fun, cov_fun_per, sigma, rho, nu, p)

        V, f, bc = PDE_solver.setup_PDE(m, pol_degree=pol_degree, a=a)

        # MC loop - sample k and solve PDE for current k
        for i in tqdm(range(N)):
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

    return Ms, avg_times, gamma, C_gamma

def main():
    # choose RHS constant for the ODE
    a_val = 1
    # polynomial degree for computing FEM approximations
    pol_degree_val = 1
    # variance of random field
    sigma = 1
    # correlation length of random field
    rho = 0.1
    # smoothness parameters
    nu = 1.5
    # mesh size for coarsest level
    m_0_val = 4
    # initial padding in circulant embedding matrix
    J1_val = J2_val = 0
    # norm to be used in covariance function of random field
    p_val = 2

    M_vec, avg_times_vec, gamma_val, C_gamma = compute_gamma(N=2, a=a_val, pol_degree=pol_degree_val, cov_fun=cov_functions.Matern_cov, cov_fun_per=periodisation_smooth.periodic_cov_fun, rho=rho, sigma=sigma, nu=nu, p=p_val)

    print(gamma_val)
    print(C_gamma)

    np.save('./data/avg_times_vec_03_15.npy', avg_times_vec, allow_pickle=True)

if __name__ == "__main__":
    main()