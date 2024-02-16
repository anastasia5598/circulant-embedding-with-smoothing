import numpy as np
import math
import scipy.special

def exponential_cov(x, sigma, rho, nu=0.5, p=1):
    ''' 
    Function for computing the covariance function of the random field at a given point in the domain, namely r(x) = sigma**2 * e^(-||x||_p / rho), where x in R^d.

    :param x: the point at which to compute the value of the covariance function.
    :param sigma: variance of random field.
    :param rho: correlation length of random field.
    :param p: the norm to use in the variance function.

    :return: 
        the value of the covariance function at x.
    '''
    # the value of the covariance function at x, using formula in docstring.
    return sigma**2 * np.exp(-np.linalg.norm(x, p) / rho)

def exponential_cov_1D(x, sigma, rho, nu=0.5, p=1):
    ''' 
    Function for computing the covariance function of the random field at a given point in the domain, namely r(x) = sigma**2 * e^(-||x||_p / rho), where x in R.

    :param x: the point at which to compute the value of the covariance function.
    :param sigma: variance of random field.
    :param rho: correlation length of random field.
    :param p: the norm to use in the variance function.

    :return: 
        the value of the covariance function at x.
    '''
    # the value of the covariance function at x, using formula in docstring.
    return sigma**2 * np.exp(-np.abs(x) / rho)

def Matern_cov(x, sigma, rho, nu, p=2):
    ''' 
    Function for computing the Mat\'ern covariance function of the random field at a given point in a d-dimensional domain.

    :param x: the point at which to compute the value of the covariance function.
    :param sigma: variance of random field.
    :param rho: correlation length of random field.
    :param nu: the smoothness exponent of the random field.

    :return: 
        the value of the covariance function at x.
    '''
    gamma_coeff = 2**(1-nu)/math.gamma(nu)

    argument = np.sqrt(2*nu)*np.linalg.norm(x, p)/rho

    kappa_coeff = scipy.special.kv(nu, argument) # this has a singularity at 0

    return sigma**2 * gamma_coeff * argument**nu * kappa_coeff

def Matern_cov_1D(x, sigma, rho, nu=2, p=2):
    ''' 
    Function for computing the Mat\'ern covariance function of the random field at a given point in a one-dimensional domain.

    :param x: the point at which to compute the value of the covariance function.
    :param sigma: variance of random field.
    :param rho: correlation length of random field.
    :param nu: the smoothness exponent of the random field.

    :return: 
        the value of the covariance function at x.
    '''
    
    gamma_coeff = 2**(1-nu)/math.gamma(nu)

    argument = np.sqrt(2*nu)*np.abs(x)/rho

    kappa_coeff = scipy.special.kv(nu, argument) # this has a singularity at 0

    return sigma**2 * gamma_coeff * argument**nu * kappa_coeff
