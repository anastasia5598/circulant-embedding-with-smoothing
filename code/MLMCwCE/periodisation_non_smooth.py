import numpy as np

def x_periodisation(x, l):
    '''
    Non-smooth periodisation function on [-l, l] as defined in Bachmayr et al. (2020).

    Input:
        x (float): point at which to do the peoridisation;
        l (float): dimension of periodisation interval.

    Output:
        The value of the periodic function at the input point.
    '''
    lower_bound = int(np.ceil(-1/2 - x/(2*l))) # for n s.t x + 2*l*n in [-l,l]
    upper_bound = int(np.ceil(1/2 - x/(2*l))) # for n s.t x + 2*l*n in [-l,l]

    n = np.arange(lower_bound, upper_bound)[0] # n's s.t x + 2*l*n in [-l,l]

    xs = x + 2*l*n # points in [-l, l]

    # return np.array([xs]) - for 1D
    return xs

x_periodisation = np.vectorize(x_periodisation)

def phi(x):
    '''
    Periodisation function which defines whether periodisation is smooth or non-smooth. For non-smooth periodisation, phi is just the characteristic function.
    '''
    return 1

phi = np.vectorize(phi)

def periodic_cov_fun(x, l, cov_fun, sigma, rho, nu, p):
    '''
    Computes the non-smooth periodic extension of a covariance function rho, as defined in Bachmayr et al. (2020).

    Input:
        x (ndarray): d-dimensional input at which to compute rho^ext.
        l (float): the size of periodisation domain [-l,l].
        cov_fun (fun): covariance function to extend.
        sigma (float): variance of covariance function.
        rho (float): correlation length of covariance function.
        nu (float): smoothness parameter of covariance function.
        p (int): norm order for covariance function.

    Output:
        cov (float): values of rho^ext at x.
    '''

    xs = x_periodisation(x,l) # find x + 2*l*n in [-l, l]
    phis = phi(np.linalg.norm(xs, np.inf, axis=0)) # apply periodisation

    cov_non_per = np.apply_along_axis(cov_fun, 0, xs, sigma, rho, nu, p) # compute covariance function at new xs

    cov = np.sum(phis*cov_non_per) # compute periodic extension

    return cov

# cov_fun_periodisation = np.vectorize(cov_fun_periodisation) - for 1D