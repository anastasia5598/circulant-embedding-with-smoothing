# circulant-embedding-with-smoothing
This is the authors' code for the paper https://arxiv.org/abs/2306.13493, in which we propose the smoothed circulant embedding method.

Note that the code provided is written in Python and requires the `FEniCS` (https://fenicsproject.org/ - using `DOLFIN`, not the newer `DOLFINx`) and `pyFFTW` (https://pypi.org/project/pyFFTW/) libraries to be installed. 

The main other python library we use are `NumPy` (https://numpy.org/) and `time` (https://docs.python.org/3/library/time.html) to get an estimate for the running time of the algorithms. 

For the Mat\'ern covariance function, we also make use of the `scipy.special.kv` function (https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kv.html) to implement the modified Bessel functions.

To track the progress of the Monte Carlo loops we use the `tqdm` library (https://tqdm.github.io/). This is commented out by default, so if you want to use it, install the library and uncomment the corresponding lines.

We provide three implementations: Monte Carlo with Circulant Embedding (MCwCE), Multilevel Monte Carlo with Circulant Embedding (MLMCwCE) and Multilevel Monte Carlo with Smoothed Circulant Embedding (MLMCwSCE). The scripts in each of these folders make use of the helper modules in utils. To avoid problems with the path when running the code, we added these helper functions in each individual folders as well.

We would like to note that the coded provided is not optimised. There are many improvement suggestions, which we have implemented in practice, but not included here. This is because we wanted the code to be easy to run. For example, one could:
* parallelise the Monte Carlo loops;
* re-use the samples generated for estimating the parameters alpha and beta in the MC and MLMC estimators;
* re-use the parameters alpha, beta and gamma across MC and MLMC with and without smoothing, as these are problem dependent, rather than algorithm dependent;
* re-use the samples generated during the simulation for a given epsilon;
* skip the initial step of estimating the variance for the number of samples bu using the pre-computed samples for alpha and beta.

We also note that this code is easy to adapt for a different problem. One only needs to change the PDE_solver module with the necessary code to solve the desired model.