"""
Tahmid F. Mehdi
Moses Lab, University of Toronto
Helper functions for Gibbs Sampler
January 31, 2018

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
import pandas as pd
from .distributions import *


def sample_parameters(hypers, n_samples):
    """
    Produces a list of n_samples parameters

    :param hyperparameters hypers: hyperparameters for distributions
    :param int>=1 n_samples: number of parameters to generate
    :return: n_samples parameters
    :rtype: list
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
    tau = list(map(lambda col: np.random.gamma(a[col], 1/b[col], n_samples), range(n_cts)))
    mu = list(map(lambda col: np.random.normal(nu[col], 1/np.sqrt(rho[col]*tau[col]), n_samples), range(n_cts)))
    p = list(map(lambda col: np.random.beta(gamma[col], delta[col], n_samples), range(n_bin)))
    lam = list(map(lambda col: np.random.gamma(eps[col], 1/zeta[col], n_samples), range(n_ord)))

    par_dicts = list(map(lambda sample: dict(mu=[mi[sample] for mi in mu], tau=[ti[sample] for ti in tau],
                                             p=[pi[sample] for pi in p],
                                             lam=[li[sample] for li in lam]), range(n_samples)))
    return par_dicts


def posterior_hyperparameters(data, hypers):
    """
    Produces posterior hyperparameters given data

    :param data: the data
    :type data: pd.DataFrame or pd.Series
    :param hyperparameters hypers: hyperparameters of distributions
    :return: posterior hyperparameters
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
    if isinstance(data, pd.DataFrame):
        m = len(data)
        x_bar = data.mean(axis=0)

        post_nu = list(map(lambda col: (rho[col]*nu[col]+m*x_bar[col])/(rho[col]+m), range(n_cts)))
        post_rho = list(map(lambda col: rho[col]+m, range(n_cts)))
        post_a = list(map(lambda col: a[col]+m/2, range(n_cts)))
        post_b = list(map(lambda col: b[col]+0.5*np.sum((data.iloc[:, col]-x_bar[col])**2)
                                      + (m*rho[col]*(x_bar[col]-nu[col])**2)/(2*(rho[col]+m)), range(n_cts)))

        post_gamma = list(map(lambda col: gamma[col]+np.sum(data.iloc[:, n_cts+col]), range(n_bin)))
        post_delta = list(map(lambda col: delta[col]+m-np.sum(data.iloc[:, n_cts+col]), range(n_bin)))

        post_eps = list(map(lambda col: eps[col]+np.sum(data.iloc[:, n_cts+n_bin+col]), range(n_ord)))
        post_zeta = list(map(lambda col: zeta[col]+m, range(n_ord)))
    else:
        m = 1
        x_bar = data

        post_nu = list(map(lambda col: (rho[col]*nu[col]+m*x_bar[col])/(rho[col]+m), range(n_cts)))
        post_rho = list(map(lambda col: rho[col]+m, range(n_cts)))
        post_a = list(map(lambda col: a[col]+m/2, range(n_cts)))
        post_b = list(map(lambda col: b[col]+0.5*(data[col]-x_bar[col])**2
                                      + (m*rho[col]*(x_bar[col]-nu[col])**2)/(2*(rho[col]+m)), range(n_cts)))

        post_gamma = list(map(lambda col: gamma[col]+data[n_cts+col], range(n_bin)))
        post_delta = list(map(lambda col: delta[col]+m-data[n_cts+col], range(n_bin)))

        post_eps = list(map(lambda col: eps[col]+data[n_cts+n_bin+col], range(n_ord)))
        post_zeta = list(map(lambda col: zeta[col]+m, range(n_ord)))

    return dict(nu=post_nu, rho=post_rho, a=post_a, b=post_b,
                gamma=post_gamma, delta=post_delta, eps=post_eps, zeta=post_zeta)


def likelihood(data, theta):
    """
    Calculates likelihood

    :param data: the data
    :type data: pd.DataFrame or pd.Series
    :param parameters theta: parameters of the distributions
    :return: P(data|theta)
    :rtype: 0<=float<=1
    """

    mu = theta['mu']
    tau = theta['tau']
    p = theta['p']
    lam = theta['lam']
    n_cts = len(mu)
    n_bin = len(p)
    n_ord = len(lam)
    product = 1
    if isinstance(data, pd.DataFrame):
        for i in range(len(data)):
            for G in range(n_cts):
                product *= normal_pdf(data.iloc[i, G], mu[G], tau[G])

            for B in range(n_bin):
                product *= bernoulli_pmf(data.iloc[i, n_cts+B], p[B])

            for P in range(n_ord):
                product *= poisson_pmf(data.iloc[i, n_cts+n_bin+P], lam[P])

    else:
        for G in range(n_cts):
            product *= normal_pdf(data[G], mu[G], tau[G])

        for B in range(n_bin):
            product *= bernoulli_pmf(data[n_cts+B], p[B])

        for P in range(n_ord):
            product *= poisson_pmf(data[n_cts+n_bin+P], lam[P])

    return product


def assign_cluster(prob_dict):
    """
    Assign a cluster based on probabilities in prob_dict

    :param dict prob_dict: mapping of clusters to probabilities
    :return: a cluster
    :rtype: int
    """

    rand = np.random.uniform(0, sum(prob_dict.values()))
    total = 0
    for clust_idx, prob in prob_dict.items():
        total += prob
        if rand <= total:
            return clust_idx
    assert False, 'Error: No Cluster Assigned'


def integrate(data, hypers):
    """
    Compute integral of P(x|parameters)P(parameters) over parameter space

    :param data: the data
    :type data: pd.DataFrame or pd.Series
    :param hyperparameters hypers: hyperparameters of distributions
    :return: integral of P(x|parameters)P(parameters) over parameter space
    :rtype: 0<=float<=1
    """

    nu = hypers['nu']
    rho = hypers['rho']
    a = hypers['a']
    b = hypers['b']
    gamma = hypers['gamma']
    delta = hypers['delta']
    eps = hypers['eps']
    zeta = hypers['zeta']
    post_hypers = posterior_hyperparameters(data, hypers)
    post_nu = post_hypers['nu']
    post_rho = post_hypers['rho']
    post_a = post_hypers['a']
    post_b = post_hypers['b']
    post_gamma = post_hypers['gamma']
    post_delta = post_hypers['delta']
    post_eps = post_hypers['eps']
    post_zeta = post_hypers['zeta']
    n_cts = len(rho)
    n_bin = len(gamma)
    n_ord = len(eps)
    product = 1
    for G in range(n_cts):
        product *= np.sqrt(rho[G]/post_rho[G])*(b[G]**a[G])/(post_b[G]**post_a[G])*sp.gamma(post_a[G])/sp.gamma(a[G])

    for B in range(n_bin):
        product *= sp.beta(post_gamma[B], post_delta[B])/sp.beta(gamma[B], delta[B])

    if isinstance(data, pd.DataFrame):
        product *= (2*np.pi)**(-n_cts*len(data)/2)
        for P in range(n_ord):
            product *= (zeta[P]**eps[P])*sp.gamma(post_eps[P])/((post_zeta[P]**post_eps[P])*sp.gamma(eps[P]))
            for i in range(len(data)):
                product *= 1/math.factorial(data.iloc[i, n_cts+n_bin+P])

    else:
        product *= (2*np.pi)**(-n_cts/2)
        for P in range(n_ord):
            product *= (zeta[P]**eps[P])*sp.gamma(post_eps[P])\
                       / ((post_zeta[P]**post_eps[P])*sp.gamma(eps[P])*math.factorial(data[n_cts+n_bin+P]))

    return product


def sample_parameters_posterior(data, hypers):
    """
    Sample parameters from posterior distribution

    :param pd.DataFrame data: data
    :param hyperparameters hypers: hyperparameters of distributions
    :return: parameters from posterior
    :rtype: parameters
    """

    if len(data) > 0:
        post_hyperparameters = posterior_hyperparameters(data, hypers)
        parameters = sample_parameters(post_hyperparameters, 1)[0]
    else:
        parameters = None

    return parameters
