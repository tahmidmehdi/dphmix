"""
Tahmid F. Mehdi
Moses Lab, University of Toronto
Unsupervised and Constrained Dirichlet Heterogeneous Mixtures of Exponential Families with Variational Inference
August 9, 2018

References:
DM Blei & MI Jordan. (2006). Variational Inference for Dirichlet Process Mixtures. Bayesian Analysis, 1(1), 121-144.
A Vlachos, A Korhonen & Z Ghahramani. (2009). Unsupervised and Constrained Dirichlet Process Mixture Models for Verb
Clustering. Proceedings of the EACL 2009 Workshop on GEMS: GEometical Models of Natural Language Semantics, 74-82.

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
from .mfviHelpers import *
from .clusters import *
from joblib import Parallel, delayed
import pandas as pd
from .utils import *


class VariationalDPHM:
    """
Implements Unsupervised & Semi-supervised Mean-field Variational Inference (MFVI) with conjugate priors for Dirichlet
Process Mixture Models where the independent variates follow Normal, Bernoulli & Poisson distributions.

If x has n_cts, n_bin & n_ord continuous, binary & ordinal variates, respectively, then
hyperparameters are contained in dictionaries where keys=['nu','rho','a','b','gamma','delta','eps','zeta'] where
the values of 'nu','rho','a' & 'b' must be lists of numbers with length n_cts
the values of 'gamma' & 'delta' must be lists of numbers with length n_bin
the values of 'eps' & 'zeta' must be lists of numbers with length n_ord

Parameters are contained in dictionaries where keys=['mu','tau','p','lam'] where
the values of 'mu' & 'tau' must be lists of numbers with length n_cts
the value of 'p' must be a list of numbers with length n_bin
the value of 'lam' must be a list of numbers with length n_ord

x | parameters ~ N(mu[1],1/tau[1])..N(mu[n_cts],1/tau[n_cts])Bern(p[1])..Bern(p[n_bin])Poi(lam[1])..Poi(lam[n_ord])
where (mu[i],tau[i]) ~ NormalGamma(nu[i],rho[i],a[i],b[i]), i=1..n_cts
      p[i] ~ Beta(gamma[i],delta[i]), i=1..n_bin
      lam[i] ~ Gamma(eps[i],zeta[i]), i=1..n_ord

For MFVI, stick-break points v[i] ~ Beta(1,alpha) yield mixing weights pi[i]=v[i]*sum(j=1..i-1)(1-v[j]) for
i=1..max_clusters.
z ~ Mult(1, pi)
Add 'sb1' & 'sb2' to hyperparameters for variational posterior of v
Variational expectations are contained in dictionaries where
keys=['lntau','tau','mu','lnp','lnip','lnlam','lam','lnv','lniv'] where
the values of 'mu','tau' & 'lntau' must be lists of numbers with length n_cts
the value of 'lnp' & 'lnip' must be a list of numbers with length n_bin
the value of 'lam' & 'lnlam' must be a list of numbers with length n_ord
the values of 'lnv' & 'lniv' must be numbers
    """
    def __init__(self, alpha, iterations, max_clusters, tol=1e-3, n_jobs=1, random_state=None):
        """
        Constructs a VariationalDPHM object

        :param float>0 alpha: the alpha parameter of the Dirichlet Process distribution
        :param int>=1 iterations: maximum number of iterations
        :param int>=2 max_clusters: maximum number of clusters
        :param float>0 tol: if inference='mfvi', the algorithm stops when the difference between evidence lower bounds
        (ELBOs) in 2 consecutive iterations is < tol
        :param int>=1 n_jobs: number of cores to use
        :param random_state: the random seed for initializing clusters in MFVI
        :type random_state: int>=0 or None
        """

        assert alpha > 0, 'alpha must be positive'
        assert iterations >= 1 and isinstance(iterations, int), 'iterations must be a positive integer'
        assert max_clusters >= 2 and isinstance(max_clusters, int), 'max_clusters must be an integer at least 2'
        assert tol > 0, 'tol must be positive'
        assert n_jobs >= 1 and isinstance(n_jobs, int), 'n_jobs must be an integer at least 1'
        self.alpha = alpha
        self.iterations = iterations
        self.max_clusters = max_clusters
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.var_moments = None

    def get_params(self):
        """
        Returns a dictionary of arguments for the object

        :return: arguments for object
        :rtype: dict
        """
        return dict(alpha=self.alpha, iterations=self.iterations, max_clusters=self.max_clusters, tol=self.tol,
                    n_jobs=self.n_jobs, random_state=self.random_state)

    def __repr__(self):
        return "VariationalDPHM(alpha={}, iterations={}, max_clusters={}, tol={}, n_jobs={}, random_state={})"\
            .format(self.alpha, self.iterations, self.max_clusters, self.tol, self.n_jobs, self.random_state)

    def __str__(self):
        return "VariationalDPHM(alpha={}, iterations={}, max_clusters={}, tol={}, n_jobs={}, random_state={})"\
            .format(self.alpha, self.iterations, self.max_clusters, self.tol, self.n_jobs, self.random_state)

    def _mfvi(self, X, hyperparameters, ml=[]):
        """
        Clusters X with variational inference given must-link constraints ml

        :param pd.DataFrame X: data
        :param hyperparameters hyperparameters: hyperparameters for distributions
        :param list ml: each element is a list of indices of points that must be in the same cluster
        :return: cluster assignment for each observation, variational parameters and moments, and final ELBO
        :rtype: MFVICluster
        """

        n = len(X)
        n_cts = len(hyperparameters['nu'])
        n_bin = len(hyperparameters['gamma'])
        ml_flat = [idx for link in ml for idx in link]
        np.random.seed(self.random_state)  # set random seed
        # initialize expectation matrix. Ez[i,j] is the probability observation i is in cluster j
        Ez = np.zeros((n, self.max_clusters))
        Ez[np.arange(n), np.random.randint(0, self.max_clusters, n)] = 1
        # stick-break points ~ Beta(1,alpha)
        hyperparameters['sb1'] = 1
        hyperparameters['sb2'] = self.alpha
        # initialize hyperparameters & parameters for each cluster
        phi_hyper = [hyperparameters]*self.max_clusters
        phi_par = [update_parameters(hyperparameters)]*self.max_clusters
        # calculate factorials for Poisson variates
        lnFactorials = X.apply(lambda row: [math.log(math.factorial(xi)) for xi in row[(n_cts+n_bin):]], axis=1)
        ELBO = float('-inf')  # initialize ELBO
        for i in range(self.iterations):
            print("Running iteration %s ... " % str(i + 1), end="")
            prevELBO = ELBO
            # update hyperparameters
            phi_hyper = Parallel(n_jobs=self.n_jobs)(delayed(update_hyperparameters)
                                                     (df=X, hypers=hyperparameters, Ez=Ez, k=k, alpha=self.alpha,
                                                      e_mu=phi_par[k]['mu']) for k in range(self.max_clusters))
            # update parameters
            phi_par = Parallel(n_jobs=self.n_jobs)(delayed(update_parameters)(hypers=phi_hyper[k])
                                                   for k in range(self.max_clusters))
            # update E(z_nk) for all n,k
            Ez = Parallel(n_jobs=self.n_jobs)(delayed(update_expectation)(x=X.iloc[idx], lnFactorial=lnFactorials[idx],
                                                                          pars=phi_par) for idx in range(n))
            Ez = np.array(Ez)
            # calculate joint probabilities for must-link data
            for link in ml:
                Ez[link, :] = np.sum(Ez[link, :], axis=0)
            # extract E[ln(1-v)] for each cluster and lnP(c=t|v) for each t=1..max_clusters
            lnivs = [phi_par[t]['lniv'] for t in range(self.max_clusters)]
            lnPc = [phi_par[t]['lnv']+np.sum(lnivs[:t]) for t in range(self.max_clusters)]
            # Add row vector lnP(c=[1..max_clusters]|v) to each row of Ez
            Ez = Ez + lnPc
            # the exp-normalize trick for preventing underflow
            rowMax = np.max(Ez, axis=1).reshape((n, 1))
            Ez = np.exp(Ez - rowMax)
            # normalize rows of Ez
            rowSums = Ez.sum(axis=1)[:, None]  # sum probabilities for each observation
            # if all probabilities for an observation are below machine epsilon then it won't be assigned a cluster
            assert 0 not in rowSums, \
                str(list(rowSums).count(0))+' observations could not be assigned to a cluster. Increase max_clusters'
            Ez = Ez/rowSums
            # create a matrix of cluster probabilities without duplicated rows for must-link constraints
            c_mat = np.delete(Ez, ml_flat, axis=0)
            for link in ml:
                c_mat = np.append(c_mat, [Ez[link[0], :]], axis=0)
            # calculate ELBO & gain
            ELBO = elbo(df=X, pars=phi_par,var_hypers=phi_hyper,prior_hypers=hyperparameters, e_z=Ez,
                        c_mat=c_mat, lnPc=lnPc, alpha=self.alpha, cores=self.n_jobs)
            ELBOgain = ELBO - prevELBO
            print("ELBO: %s ... gained %s" % (str(ELBO), str(ELBOgain)))
            if ELBOgain < self.tol:
                print("ELBO converged!")
                break
        # assign to each observation, the cluster it's most likely to belong to from the expectation matrix
        c = np.array(Ez.argmax(axis=1)).reshape(n)
        # map cluster indices so they're all consecutive integers
        uniq_c = np.unique(c)
        # filter hyperparameters & expectations
        var_hyper = [phi_hyper[j] for j in uniq_c]
        self.var_moments = [phi_par[j] for j in uniq_c]
        # a mapping to help read the E(z_nk) matrix
        cluster_map = {}
        for clust in uniq_c:
            cluster_map[clust] = np.where(uniq_c == clust)[0][0]

        clusters = list(map(lambda clust_idx: cluster_map[clust_idx], c))
        solution = MFVICluster(clusters, var_hyper, self.var_moments, Ez, cluster_map, ELBO)
        if solution.n_clusters == self.max_clusters:
            print("Warning: max_clusters reached, you may need to cluster again with a higher max_clusters.")
        print("Done")
        return solution

    def fit_predict(self, X, ml=[], hyperparameters=None):
        """
        Fit the model to X & predict clusters for each observation (row) in X given must-link constraints ml

        :param pd.DataFrame X: data
        :param list ml: each element is a list of indices of points that must be in the same cluster
        :param hyperparameters: hyperparameters of distributions for the variates
        :type hyperparameters: hyperparameters or None
        :return: the clustering solution
        :rtype: MFVICluster
        """

        assert isinstance(X, pd.DataFrame), 'X must be a pandas.DataFrame'
        ml_flat = [idx for link in ml for idx in link]
        assert not(is_duplicate(ml_flat)), 'Integers in ml are not unique'
        assert (0 <= np.array(ml_flat)).all() and (np.array(ml_flat) < len(X)).all() \
            and all([isinstance(item, int) for item in ml_flat]), 'Elements of ml must be integers in [0,len(X))'
        # sort features so continuous variables come first, then binary, followed by non-negative discrete
        X, n_cts, n_bin, n_ord = sort_features(X)
        if hyperparameters is None:
            # set prior hyperparameters
            nu = np.mean(X[X.columns[:n_cts]])
            rho = [1]*n_cts
            a = [1]*n_cts
            b = (np.var(X[X.columns[:n_cts]]))
            gamma = [2]*n_bin
            delta = (2-2*np.mean(X[X.columns[n_cts:(n_cts+n_bin)]]))/np.mean(X[X.columns[n_cts:(n_cts+n_bin)]])
            eps = [1]*n_ord
            zeta = 1/np.mean(X[X.columns[(n_cts+n_bin):]])
            hyperparameters = dict(nu=nu, rho=rho, a=a, b=b, gamma=gamma, delta=delta, eps=eps, zeta=zeta)
        else:  # check lengths of hyperparameters
            assert isinstance(hyperparameters, dict), 'hyperparameters must be a dictionary'
            assert 'nu' in hyperparameters and 'rho' in hyperparameters and 'a' in hyperparameters \
                   and 'b' in hyperparameters and 'gamma' in hyperparameters and 'delta' in hyperparameters \
                   and 'eps' in hyperparameters and 'zeta' in hyperparameters, \
                'hyperparameters needs keys "nu", "rho", "a", "b", "gamma", "delta", "eps" & "zeta"'
            assert len(hyperparameters['nu']) == n_cts and len(hyperparameters['rho']) == n_cts \
                   and len(hyperparameters['a']) == n_cts and len(hyperparameters['b']) == n_cts, \
                'Number of hyperparameters for continuous variates do not match the number of continuous variates'
            assert len(hyperparameters['gamma']) == n_bin and len(hyperparameters['delta']) == n_bin, \
                'Number of hyperparameters for binary variates do not match the number of binary variates'
            assert len(hyperparameters['eps']) == n_ord and len(hyperparameters['zeta']) == n_ord, \
                'Number of hyperparameters for ordinal variates do not match the number of ordinal variates'

        # perform MFVI
        clust_soln = self._mfvi(X, hyperparameters, ml)

        return clust_soln

    def predict(self, X):
        """
        Predict the cluster of each observation (row) in X

        :param pd.DataFrame X: data
        :return: the clustering solution
        :rtype: np.array, np.ndarray
        """

        assert self.var_moments is not None, 'Model has not been fit'
        # sort features so continuous variables come first, then binary, followed by non-negative discrete
        X, n_cts_data, n_bin_data, n_ord_data = sort_features(X)
        n = len(X)
        n_clusters = len(self.var_moments)
        n_cts = len(self.var_moments[0]['mu'])
        n_bin = len(self.var_moments[0]['lnp'])
        n_ord = len(self.var_moments[0]['lam'])
        assert n_cts+n_bin+n_ord == len(X.columns), 'The features in X do not match the features of the model'
        # log factorials of Poisson features
        lnFactorials = X.apply(lambda row: [math.log(math.factorial(xi)) for xi in row[(n_cts+n_bin):]], axis=1)
        # update E(z_nk) for all n,k
        Ez = Parallel(n_jobs=self.n_jobs)(delayed(update_expectation)(x=X.iloc[idx], lnFactorial=lnFactorials[idx],
                                                                      pars=self.var_moments) for idx in range(n))
        Ez = np.array(Ez)
        # extract E[ln(1-v)] for each cluster and lnP(c=t|v) for each t=1..max_clusters
        lnivs = [self.var_moments[t]['lniv'] for t in range(n_clusters)]
        lnPc = [self.var_moments[t]['lnv']+np.sum(lnivs[:t]) for t in range(n_clusters)]
        # Add row vector lnP(c=[1..max_clusters]|v) to each row of Ez
        Ez = Ez + lnPc
        # the exp-normalize trick for preventing underflow
        rowMax = np.max(Ez, axis=1).reshape((n, 1))
        Ez = np.exp(Ez - rowMax)
        # normalize rows of Ez
        rowSums = Ez.sum(axis=1)[:, None]  # sum probabilities for each observation
        # if all probabilities for an observation are below machine epsilon then it won't be assigned a cluster
        assert 0 not in rowSums, \
            str(list(rowSums).count(0)) + ' observations could not be assigned to a cluster. Increase max_clusters'
        Ez = Ez/rowSums
        # pick most probable cluster for each observation
        c = np.array(Ez.argmax(axis=1)).reshape(n)
        return c, Ez

