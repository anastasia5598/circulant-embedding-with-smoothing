import numpy as np

def cut_off(x):
    '''
    Cut-off function for smooth periodisation, see Bachmayr et al. (2020).
    '''
    return np.exp(-1/x) if x > 0 else 0

cut_off = np.vectorize(cut_off)

def phi(x, kappa):
    '''
    Periodisation function which defines whether periodisation is smooth or non-smooth. For smooth periodisation, phi is as in Bachmayr et al. (2020).
    '''
    numerator = cut_off((kappa - np.abs(x))/(kappa-1))
    denominator = numerator + cut_off((np.abs(x)-1)/(kappa-1))

    return numerator/denominator

phi = np.vectorize(phi)

def x_periodisation(x, l):
    '''
    Smooth periodisation function on [1-2l, 2l-1] as defined in Bachmayr et al. (2020) in Eq. (5.1).

    Input:
        x (float): point at which to do the peoridisation;
        l (float): dimension of periodisation interval.

    Output:
        The value of the periodic function at the input point.
    '''
    kappa = 2*l-1

    lower_bound = int(np.ceil((-kappa - x) / (2*l)))
    upper_bound = int(np.ceil((kappa - x) / (2*l)))

    n = np.arange(lower_bound, upper_bound)

    xs = x + 2*l*n
    
    return xs

x_periodisation = np.vectorize(x_periodisation, otypes=[np.ndarray])

def periodic_cov_fun(x, l, cov_fun, sigma, rho, nu, p):
    '''
    Computes the smooth periodic extension of a covariance function rho, as defined in Bachmayr et al. (2020).

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
    ys = x_periodisation(x,l) # find x + 2*l*n in [-l, l]
    d = len(ys)

    shapes = np.zeros(d)
    for j in range(d):
        shapes[j] = len(ys[j])
    
    xs = np.tile(np.repeat(ys[0], 1), int(np.prod(shapes)/shapes[0]))
    for i in range(d-2):
        xs = np.vstack((xs, np.tile(np.repeat(ys[i+1], int(np.prod(shapes[:i+1]))), int(np.prod(shapes[i+2:])))))
    xs = np.vstack((xs, np.tile(np.repeat(ys[d-1], int(np.prod(shapes)/shapes[d-1])), 1)))

    kappa = 2*l-1
    phis = phi(np.linalg.norm(xs, np.inf, axis=0), kappa)

    cov_non_per = np.apply_along_axis(cov_fun, 0, xs, sigma, rho, nu, p)

    cov = np.sum(phis*cov_non_per)

    return cov

# cov_fun_per = np.vectorize(cov_fun_per) - for 1D