import numpy as np
from pyfftw import *
import cov_functions
import periodisation_non_smooth
import periodisation_smooth

def build_cov_row(m, m0, d, cov_fun, cov_fun_per, sigma, rho, nu, p):
    ''' 
    Function for computing the 1st row in the circulant embedding matrix.

    :param m: number of points in each direction, including periodisation.
    :param m0: initial mesh size for discretisation in each direction.
    :param d: dimension of problem (number of coordinates).
    :param cov_fun: the function for which we compute the circulant embedding.
    :param cov_fun_per: periodisation function (smooth or non-smooth).
    :param sigma: variance of random field.
    :param rho: correlation length of random field.
    :param nu: smoothness parameter of random field.
    :param p: the norm to use in the variance function.

    :return: 
        row - the first row in the circulant embedding matrix.
    '''
    # dimension of peoridisation
    l = m/m0

    # empty array for generating points at which to compute covariance function
    pts = np.zeros(((2*m)**d, d))

    # padding
    for i in range(d):
        s = (2*m)**i
        pts[:, i] = np.array([x//s for x in range((2*m)*s)]*(2*m)**(d-i-1))
 
    # normalise points
    pts = pts/m0
    # there is a singularity at 0, so we add 1 for the [0,0] point
    row = np.ones((2*m)**d)
    # compute covariance function at each such point
    row[1:] = np.apply_along_axis(cov_fun_per, 1, pts[1:], l, cov_fun, sigma, rho, nu, p).reshape(((2*m)**d-1))
    # row[1:] = cov_fun_per(pts[1:], l, cov_fun, per_fun, sigma, rho, nu, p).reshape((2*m-1)) - for 1D

    return row

def find_cov_eigenvalues(m, m0, d, cov_fun, cov_fun_per, sigma, rho, nu, p):
    ''' 
    Function for computing the eigenvalues of the circulant embedding matrix, using the fast Fourier transform of its first row/column. If the eigenvalues are not real and positive, the padding is increased, and the eigenvalues re-computed, until they are real and positive.

    :param m: number of points in each direction, including periodisation.
    :param m0: initial mesh size for discretisation in each direction.
    :param d: dimension of problem (number of coordinates).
    :param cov_fun: the function for which we compute the circulant embedding.
    :param cov_fun_per: periodisation function.
    :param sigma: variance of random field.
    :param rho: correlation length of random field.
    :param nu: smoothness parameter of random field.
    :param p: the norm to use in the variance function.

    :return: 
        egnv - eigenvalues of circuland embedding matrix.
        s - dimension of circulant embedding matrix.
    '''
    # get the first row/column in the circulant embedding matrix
    if m==m0:
        c = build_cov_row(m,m0,d,cov_fun, periodisation_non_smooth.periodic_cov_fun, sigma, rho, nu, p)
    else:
        c = build_cov_row(m,m0,d,cov_fun, cov_fun_per, sigma, rho, nu, p)

    # compute dimension of circulant embedding matrix
    s = (2*m)**d
    fft_shape = tuple([2*m]*d)

    # find eigenvalues of circulant embeding matrix by computing the n-dimensional fast Fourier transform of its first column
    egnv = interfaces.scipy_fft.fftn(c.reshape(fft_shape)).reshape(s)

    # increasing padding until eigenvalues are real and positive
    while (any(np.round(np.real(egnv), 12) < 0) or any(np.round(np.imag(egnv), 7) != 0)):
        m += 1
        # print(m)
        s = (2*m)**d
        fft_shape = tuple([2*m]*d)
        c = build_cov_row(m,m0,d,cov_fun,cov_fun_per, sigma, rho, nu, p)
        egnv = interfaces.scipy_fft.fftn(c.reshape(fft_shape)).reshape(s)

    # check that eigenvalues are indeed real and positive
    assert(all(np.round(np.imag(egnv), 10) == 0))
    egnv = np.real(egnv)
    assert(all(np.round(egnv, 10) > 0))

    return egnv, m

# below is an example of computing the circulant embedding eigenvalues
# uncomment to run
# if __name__ == "__main__":
#     m = 32
#     d = 2
#     rho = 0.01
#     nu = 0.5
#     p = 1
#     egnv, m = find_cov_eigenvalues(m, m, d, cov_functions.exponential_cov, periodisation_smooth.periodic_cov_fun, sigma=1, rho=rho, nu=nu, p=p)