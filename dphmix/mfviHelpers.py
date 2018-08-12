"""
Tahmid F. Mehdi
Moses Lab, University of Toronto
Helper functions for Variational Inference
July 26, 2018

Copyright 2018 Tahmid Mehdi
This file is part of dphmix.

dphmix is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

dphmix is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with dphmix.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import division
from .distributions import *
from joblib import Parallel, delayed
import scipy.special as sp


def update_hyperparameters(df, hypers, Ez, k, alpha, e_mu):
    """
    Update the hyperparameters of all distributions with their posterior values

    :param pd.DataFrame df: data
    :param hyperparameters hypers: hyperparameters of distributions
    :param np.ndarray Ez: probability matrix
    :param int>=0 k: cluster index
    :param float>0 alpha: alpha parameter of Dirichlet Process
    :param float e_mu: E(mu)
    :return: updated hyperparameters
    :rtype: hyperparameters
    """

    nu = hypers['nu']
    rho = hypers['rho']
    a = hypers['a']
    b = hypers['b']
    gamma = hypers['gamma']
    delta = hypers['delta']
    eps = hypers['eps']
    zeta = hypers['zeta']
    n_cts = len(nu)
    n_bin = len(gamma)
    n_ord = len(eps)

    post_rho = list(map(lambda col: rho[col]+np.sum(Ez[:, k]), range(n_cts)))
    post_nu = list(map(lambda col: (1/post_rho[col])*(rho[col]*nu[col]+np.sum(Ez[:, k]*df.iloc[:, col].values)),
                       range(n_cts)))
    post_a = list(map(lambda col: a[col]+0.5*np.sum(Ez[:, k]), range(n_cts)))
    post_b = list(map(lambda col: b[col]+0.5*(np.sum(Ez[:, k]*(df.iloc[:, col].values-e_mu[col])**2)
                                              + rho[col]*(e_mu[col]-nu[col])**2), range(n_cts)))
    post_gamma = list(map(lambda col: gamma[col]+np.sum(Ez[:, k]*df.iloc[:, n_cts+col].values), range(n_bin)))
    post_delta = list(map(lambda col: delta[col]+np.sum(Ez[:, k]*(1-df.iloc[:, n_cts+col].values)), range(n_bin)))
    post_eps = list(map(lambda col: eps[col]+np.sum(Ez[:, k]*df.iloc[:, n_cts+n_bin+col].values), range(n_ord)))
    post_zeta = list(map(lambda col: zeta[col]+np.sum(Ez[:, k]), range(n_ord)))
    u1 = 1+np.sum(Ez[:, k])
    u2 = alpha+np.sum(Ez[:, (k+1):].sum(axis=0))
    return dict(nu=post_nu, rho=post_rho, a=post_a, b=post_b,
                gamma=post_gamma, delta=post_delta, eps=post_eps, zeta=post_zeta,
                sb1=u1, sb2=u2)


def update_parameters(hypers):
    """
    Update the parameters of all distributions with their expectations

    :param hyperparameters hypers: hyperparameters of distributions
    :return: moments of parameters
    :rtype: variational expectations
    """

    nu = hypers['nu']
    rho = hypers['rho']
    a = hypers['a']
    b = hypers['b']
    gamma = hypers['gamma']
    delta = hypers['delta']
    eps = hypers['eps']
    zeta = hypers['zeta']
    sb1 = hypers['sb1']
    sb2 = hypers['sb2']
    n_cts = len(nu)
    n_bin = len(gamma)
    n_ord = len(eps)

    e_lntau = np.array(list(map(lambda col: sp.digamma(a[col])-np.log(b[col]), range(n_cts))))
    e_tau = np.array(list(map(lambda col: a[col]/b[col], range(n_cts))))
    e_mu = np.array(list(map(lambda col: nu[col], range(n_cts))))
    e_lnp = np.array(list(map(lambda col: sp.digamma(gamma[col])-sp.digamma(gamma[col]+delta[col]), range(n_bin))))
    e_lnip = np.array(list(map(lambda col: sp.digamma(delta[col])-sp.digamma(gamma[col]+delta[col]), range(n_bin))))
    e_lnlam = np.array(list(map(lambda col: sp.digamma(eps[col])-np.log(zeta[col]), range(n_ord))))
    e_lam = np.array(list(map(lambda col: eps[col]/zeta[col], range(n_ord))))
    e_lnv = sp.digamma(sb1)-sp.digamma(sb1+sb2)
    e_lniv = sp.digamma(sb2)-sp.digamma(sb1+sb2)

    return dict(lntau=e_lntau,tau=e_tau,mu=e_mu, lnp=e_lnp,lnip=e_lnip, lnlam=e_lnlam,lam=e_lam, lnv=e_lnv,lniv=e_lniv)


def calc_expectations(x, lnFactorial, pars):
    """
    Helps update_expectations

    :param pd.Series x: data
    :param list lnFactorial: each element is the ln of the factorial of a poisson variate
    :param variational expectations pars: parameters for updating probability matrix
    :return: an element for the probability matrix before exponentiation
    :rtype: float
    """

    e_lntau = pars['lntau']
    e_tau = pars['tau']
    e_mu = pars['mu']
    e_lnp = pars['lnp']
    e_lnip = pars['lnip']
    e_lnlam = pars['lnlam']
    e_lam = pars['lam']

    n_cts = len(e_mu)
    n_bin = len(e_lnp)

    e = 0.5*np.sum(e_lntau-e_tau*(x[:n_cts]-e_mu)**2) \
        + np.sum(x[n_cts:(n_cts+n_bin)]*e_lnp+(1-x[n_cts:(n_cts+n_bin)])*e_lnip) \
        + np.sum(x[(n_cts+n_bin):]*e_lnlam-lnFactorial-e_lam)

    return e


def update_expectation(x, lnFactorial, pars):
    """
    Update expectation matrix

    :param pd.Series x: data
    :param list lnFactorial: each element is the ln of the factorial of a poisson variate
    :param list pars: each element is a variational expectation
    :return: a column of the probability matrix
    :rtype: list
    """

    max_clusters = len(pars)
    en = list(map(lambda t: calc_expectations(x, lnFactorial, pars[t]), range(max_clusters)))
    return en


def elbo_loglik_cluster(x, pars):
    """
    Calculate E[lnP(x|z,pars)] for a cluster with parameters pars

    :param pd.Series x: data
    :param variational expectations pars: parameters for calculating ELBO
    :return: part of the ELBO
    :rtype: float
    """

    e_lntau = pars['lntau']
    e_tau = pars['tau']
    e_mu = pars['mu']
    e_lnp = pars['lnp']
    e_lnip = pars['lnip']
    e_lnlam = pars['lnlam']
    e_lam = pars['lam']
    n_cts = len(e_mu)
    n_bin = len(e_lnp)
    n_ord = len(e_lam)
    sum_ln_normal = np.sum(list(map(lambda G: ln_normal(x[G], e_lntau[G], e_mu[G], e_tau[G]), range(n_cts))))
    sum_ln_bernoulli = np.sum(list(map(lambda B: ln_bernoulli(x[n_cts+B], e_lnp[B], e_lnip[B]), range(n_bin))))
    sum_ln_poisson = np.sum(list(map(lambda P: ln_poisson(x[n_cts+n_bin+P], e_lnlam[P], e_lam[P]), range(n_ord))))
    return sum_ln_normal+sum_ln_bernoulli+sum_ln_poisson


def elbo_loglik(x, pars):
    """
    Calculate E[lnP(x|z,pars)]

    :param pd.Series x: data
    :param list pars: elements are variational expectations
    :return: E[lnP(x|z,pars)] for ELBO
    :rtype: float
    """

    max_clusters = len(pars)
    sum = list(map(lambda k: elbo_loglik_cluster(x, pars[k]), range(max_clusters)))
    return sum


def elbo_prior(pars, hypers):
    """
    Calculate E[lnP(pars|hypers)]

    :param variational expectations pars: parameters for ELBO
    :param hyperparameters hypers: hyperparameters for distributions
    :return: E[lnP(pars|hypers)]
    :rtype: float
    """

    nu = hypers['nu']
    rho = hypers['rho']
    a = hypers['a']
    b = hypers['b']
    gamma = hypers['gamma']
    delta = hypers['delta']
    eps = hypers['eps']
    zeta = hypers['zeta']

    e_lntau = pars['lntau']
    e_tau = pars['tau']
    e_mu = pars['mu']
    e_lnp = pars['lnp']
    e_lnip = pars['lnip']
    e_lnlam = pars['lnlam']
    e_lam = pars['lam']
    n_cts = len(e_mu)
    n_bin = len(e_lnp)
    n_ord = len(e_lam)
    sum_ln_ng = np.sum(list(map(lambda G: ln_normalgamma(e_mu[G], e_tau[G], e_lntau[G], nu[G],rho[G], a[G],b[G]),
                                range(n_cts))))
    sum_ln_beta = np.sum(list(map(lambda B: ln_beta(e_lnp[B], e_lnip[B], gamma[B], delta[B]), range(n_bin))))
    sum_ln_gamma = np.sum(list(map(lambda P: ln_gamma(e_lam[P], e_lnlam[P], eps[P], zeta[P]), range(n_ord))))
    return sum_ln_ng+sum_ln_beta+sum_ln_gamma


def local_sum_par(pars, var_hypers):
    """
    Helps parallelize ELBO computation

    :param variational expectations pars: parameters for ELBO
    :param hyperparameters var_hypers: hyperparameters of distributions
    :return: part of ELBO
    :rtype: float
    """

    return elbo_prior(pars, var_hypers)+ln_beta(pars['lnv'], pars['lniv'], var_hypers['sb1'], var_hypers['sb2'])


def elbo(df, pars, var_hypers, prior_hypers, e_z, c_mat, lnPc, alpha, cores):
    """
    Calculate ELBO

    :param pd.DataFrame df: data
    :param variational expectations pars: parameters of ELBO
    :param hyperparameters var_hypers: variational parameters
    :param hyperparameters prior_hypers: prior hyperparameters of distributions
    :param np.ndarray e_z: probability matrix
    :param np.ndarray c_mat: cluster probability matrix
    :param list lnPc: list of lnP(c=[1..T]|v)
    :param float>0 alpha: parameter for Dirichlet Process
    :param int>=1 cores: number of cores to use
    :return: ELBO
    :rtype: float<=0
    """

    n = len(df)
    max_clusters = e_z.shape[1]
    loglik_sums = np.array(Parallel(n_jobs=cores)(delayed(elbo_loglik)(x=df.iloc[idx], pars=pars) for idx in range(n)))
    loglik = np.sum(e_z*loglik_sums)
    logprior = np.sum(Parallel(n_jobs=cores)(delayed(elbo_prior)(pars=pars[k], hypers=prior_hypers)
                                             for k in range(max_clusters)))
    c_mat_colSums = np.sum(c_mat, axis=0)
    lnpz = np.sum(c_mat_colSums*lnPc)
    lnpv = np.sum(Parallel(n_jobs=cores)(delayed(ln_beta)(ln_x=pars[k]['lnv'], ln_ix=pars[k]['lniv'], a=1, b=alpha)
                                         for k in range(max_clusters)))
    local_sum = np.sum(Parallel(n_jobs=cores)(delayed(local_sum_par)(pars=pars[k], var_hypers=var_hypers[k])
                                              for k in range(max_clusters)))
    lnq = np.sum(np.log(np.power(c_mat, c_mat)))+local_sum
    return loglik+logprior+lnpz+lnpv-lnq
