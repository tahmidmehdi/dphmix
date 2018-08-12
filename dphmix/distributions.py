"""
Tahmid F. Mehdi
Moses Lab, University of Toronto
Probability Functions for Several Exponential Families
February 5, 2018

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
import math
import numpy as np
import scipy.special as sp


def normal_pdf(x, mu, tau):
    """
    PDF of normal random variable x

    :param float x: data
    :param float mu: mean
    :param float>0 tau:  precision
    :return: probability
    :rtype: 0<=float<=1
    """

    return np.sqrt(tau/(2*np.pi))*np.exp(-(tau*(x-mu)**2)/2)


def bernoulli_pmf(x, p):
    """
    PMF of bernoulli random variable x

    :param int x: binary data. {0,1}
    :param 0<=float<=1 p: probability of success
    :return: probability
    :rtype: 0<=float<=1
    """

    return p**x*(1-p)**(1-x)


def poisson_pmf(x, lam):
    """
    PMF of poisson random variable x

    :param int>=0 x: data
    :param float>=0 lam: rate parameter
    :return: probability
    :rtype: 0<=float<=1
    """

    return lam**x*np.exp(-lam)/math.factorial(x)


def gamma_pdf(x, a, b):
    """
    PDF of gamma random variable x

    :param float>=0 x: data
    :param float>0 a: shape
    :param float>0 b: rate
    :return: probability
    :rtype: 0<=float<=1
    """

    return b**a*x**(a-1)*np.exp(-b*x)/sp.gamma(a)


def beta_pdf(x, a, b):
    """
    PDF of beta random variable x

    :param float x: data. [0,1]
    :param float>0 a: shape
    :param float>0 b: shape
    :return: probability
    :rtype: 0<=float<=1
    """

    return (x**(a-1)*(1-x)**(b-1))/sp.beta(a, b)


def ln_normal(x, ln_tau, mu, tau):
    """
    ln probability of Normal random variable x

    :param float x: data
    :param float ln_tau: ln(precision)
    :param float mu: mean
    :param float>0 tau: precision
    :return: lnP(x)
    :rtype: float<=0
    """

    return 0.5*(ln_tau-np.log(2*np.pi)-tau*(x-mu)**2)


def ln_bernoulli(x, ln_p, ln_ip):
    """
    ln probability of Bernoulli random variable x

    :param int x: data. {0,1}
    :param float<=0 ln_p: ln(p)
    :param float<=0 ln_ip: ln(1-p)
    :return: lnP(x)
    :rtype: float<=0
    """

    return x*ln_p+(1-x)*ln_ip


def ln_poisson(x, ln_lam, lam):
    """
    ln probability of Poisson random variable x

    :param int>=0 x: data
    :param float ln_lam: ln(rate)
    :param float>=0 lam: rate parameter
    :return: lnP(x)
    :rtype: float<=0
    """

    return x*ln_lam-math.log(math.factorial(x))-lam


def ln_normalgamma(x, y, ln_y, nu, rho, a, b):
    """
    ln probability of NormalGamma random variable (x,y)

    :param float x: data
    :param float>0 y: data
    :param float ln_y: ln(y)
    :param float nu: mean
    :param float>0 rho: precision
    :param float>0 a: shape
    :param float>0 b: rate
    :return: lnP(x,y)
    :rtype: float<=0
    """

    return 0.5*(math.log(rho/(2*np.pi))+ln_y-rho*y*(x-nu)**2)+a*math.log(b)-sp.gammaln(a)+(a-1)*ln_y-b*y


def ln_beta(ln_x, ln_ix, a, b):
    """
    ln probability of Beta random variable

    :param float<=0 ln_x: ln(x). x is the data
    :param float<=0 ln_ix: ln(1-x)
    :param float>0 a: shape
    :param float>0 b: shape
    :return: lnP(x)
    :rtype: float<=0
    """

    return (a-1)*ln_x+(b-1)*ln_ix-sp.betaln(a, b)


def ln_gamma(x, ln_x, a, b):
    """
    ln probability of Gamma random variable x

    :param float>=0 x: data
    :param float ln_x: ln(x)
    :param float>0 a: shape
    :param float>0 b: rate
    :return: lnP(x)
    :rtype: float<=0
    """

    return a*math.log(b)-sp.gammaln(a)+(a-1)*ln_x-b*x
