"""
Tahmid F. Mehdi
Moses Lab, University of Toronto
Unsupervised and Constrained Dirichlet Heterogeneous Mixture of Exponential Families with Gibbs Sampling
July 19, 2018

References:
RM Neal. (2000). Markov Chain Sampling Methods for Dirichlet Process Mixture Models. Journal of Computational and
Graphical Statistics, 9(2), 249-265.
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
from .gibbsHelpers import *
from .clusters import *
from collections import Counter
from itertools import count, filterfalse
from joblib import Parallel, delayed
from .utils import *


class GibbsDPHM:
    """
Implements Algorithm 2 from Neal (2000) with conjugate priors for Dirichlet Process Mixture Models where the independent
variates follow Normal, Bernoulli & Poisson distributions. It can handle must-link & can't-link constraints

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
    """
    def __init__(self, alpha, iterations, max_clusters, n_jobs=1):
        """
        Constructs a GibbsDPHM object

        :param float>0 alpha: the alpha parameter of the Dirichlet Process distribution
        :param int>=1 iterations: maximum number of iterations
        :param int>=2 max_clusters: maximum number of clusters
        :param int>=1 n_jobs: number of cores to use
        """

        assert alpha > 0, 'alpha must be positive'
        assert iterations >= 1 and isinstance(iterations, int), 'iterations must be a positive integer'
        assert max_clusters >= 2 and isinstance(max_clusters, int), 'max_clusters must be an integer at least 2'
        assert n_jobs >= 1 and isinstance(n_jobs, int), 'n_jobs must be an integer at least 1'
        self.alpha = alpha
        self.iterations = iterations
        self.max_clusters = max_clusters
        self.n_jobs = n_jobs
        self.clusters = None
        self.phi = None

    def get_params(self):
        """
        Returns a dictionary of arguments for the object

        :return: arguments for object
        :rtype: dict
        """
        return dict(alpha=self.alpha, iterations=self.iterations, max_clusters=self.max_clusters, n_jobs=self.n_jobs)

    def __repr__(self):
        return "GibbsDPHM(alpha={}, iterations={}, max_clusters={}, n_jobs={})".\
            format(self.alpha, self.iterations, self.max_clusters, self.n_jobs)

    def __str__(self):
        return "GibbsDPHM(alpha={}, iterations={}, max_clusters={}, n_jobs={})".\
            format(self.alpha, self.iterations, self.max_clusters, self.n_jobs)

    def _gibbs(self, X, hyperparameters, ml=[], cl=[]):
        """
        Clusters X with Gibbs Sampler given must-link constraints ml & can't-link constraints cl

        :param pd.DataFrame X: data
        :param hyperparameters hyperparameters: hyperparameters for distributions
        :param list ml: each element is a list of indices of points that must be in the same cluster
        :param list cl: each element is a list of indices of points that cannot be in the same cluster
        :return: cluster assignments for each observation and cluster parameters
        :rtype: GibbsCluster
        """

        n = len(X)
        ml_flat = [idx for link in ml for idx in link]
        not_ml = [i for i in range(n) if i not in ml_flat]

        # create mapping of data indices to indices of observations they cannot be clustered with
        cantLink = {}
        for link in cl:
            for idx in link:
                if idx in cantLink:
                    cantLink[idx].extend([m for m in link if m != idx])
                else:
                    cantLink[idx] = [m for m in link if m != idx]
        # create mapping of must-links to indices of observations they cannot be clustered with
        cantLink_ml = {}
        integral_ml = [0]*len(ml)
        for link in range(len(ml)):
            cantLink_ml[link] = []
            for idx in ml[link]:
                if idx in cantLink:
                    cantLink_ml[link].extend(cantLink[idx])

            integral_ml[link] = integrate(X.iloc[ml[link]], hyperparameters)

        c = np.array([0]*n)  # randomly assign clusters
        # initialize cluster parameters. If unsupervised then just initialize 1 cluster. If semi-supervised then
        # initialize clusters for each label as well as cluster 0 for unlabeled data
        phi = Parallel(n_jobs=self.n_jobs)(delayed(sample_parameters_posterior)
                                           (data=X.iloc[c == clust], hypers=hyperparameters)
                                           for clust in range(self.max_clusters))
        # calculate integrals
        integral = X.apply(lambda x: integrate(x, hyperparameters), axis=1)
        for state in range(self.iterations):
            print("Running MCMC Step %s" % str(state + 1))
            for i in not_ml:
                c_noi = np.delete(c, i, axis=0)  # clusters without X[i]
                n_noi = Counter(c_noi)  # count frequency of each cluster
                if c[i] not in c_noi:  # X[i] is the only one from its cluster
                    phi[c[i]] = None  # delete its cluster

                all_clusters = np.unique(c_noi)
                candidate_clusters = np.copy(all_clusters)
                # remove clusters of data that the ith point cannot cluster with
                if i in cantLink:
                    candidate_clusters = [clust for clust in all_clusters if clust not in map(lambda a: c[a], cantLink[i])]
                # dictionary of probabilities of each cluster
                prob_c = {}
                # set probabilities for existing clusters according to Chinese Restaurant Process
                for clust in candidate_clusters:
                    prob_c[clust] = n_noi[clust]*likelihood(X.iloc[i], phi[clust])
                # find the lowest non-negative integer that hasn't indexed a cluster
                new_cluster = next(filterfalse(all_clusters.__contains__, count(0)))
                # set probability for new cluster
                prob_c[new_cluster] = self.alpha*integral[i]
                c[i] = assign_cluster(prob_c)

                if c[i] == new_cluster:  # if X[i] was assigned the new cluster
                    assert new_cluster < self.max_clusters, 'Number of clusters exceeded max_clusters'
                    # set parameters for new cluster
                    post_hyperparameters = posterior_hyperparameters(X.iloc[i], hyperparameters)
                    phi[new_cluster] = sample_parameters(post_hyperparameters, 1)[0]
            # pass through data in must-link constraints
            for link in range(len(ml)):
                c_noi = np.delete(c, ml[link], axis=0)  # clusters without X[i]
                n_noi = Counter(c_noi)  # count frequency of each cluster

                all_clusters = np.unique(c_noi)
                candidate_clusters = np.copy(all_clusters)
                # remove clusters of points the group cannot cluster with from all_clusters
                if i in cantLink_ml:
                    candidate_clusters = [clust for clust in all_clusters if clust not in map(lambda a: c[a], cantLink_ml[link])]
                # dictionary of probabilities of each cluster
                prob_c = {}
                # set probabilities for existing clusters according to Chinese Restaurant Process
                for clust in candidate_clusters:
                    # use joint probability of data in ml[link]
                    prob_c[clust] = n_noi[clust]*likelihood(X.iloc[ml[link]], phi[clust])
                # find the lowest non-negative integer that hasn't indexed a cluster
                new_cluster = next(filterfalse(all_clusters.__contains__, count(0)))
                # set probability for new cluster
                prob_c[new_cluster] = self.alpha*integral_ml[link]
                c[ml[link]] = assign_cluster(prob_c)

                if c[ml[link][0]] == new_cluster:  # if X[i] was assigned the new cluster
                    assert new_cluster < self.max_clusters, 'Number of clusters exceeded max_clusters'
                    # set parameters for new cluster
                    post_hyperparameters = posterior_hyperparameters(X.iloc[ml[link]], hyperparameters)
                    phi[new_cluster] = sample_parameters(post_hyperparameters, 1)[0]
            # resample parameters for all clusters from the posterior
            phi = Parallel(n_jobs=self.n_jobs)(delayed(sample_parameters_posterior)
                                               (data=X.iloc[c == clust], hypers=hyperparameters)
                                               for clust in range(self.max_clusters))

        # remove Nones from phi
        self.phi = [theta for theta in phi if theta is not None]
        # map cluster indices so they're all consecutive integers
        uniq_c = np.unique(c)
        self.clusters = list(map(lambda clust: np.where(uniq_c == clust)[0][0], c))
        solution = GibbsCluster(self.clusters, self.phi)
        print("Done")
        return solution

    def fit_predict(self, X, ml=[], cl=[], hyperparameters=None):
        """
        Fits a model & predict clusters for each observation (row) in X given must-link constraints ml and can't-link
        constraints cl

        :param pd.DataFrame X: data
        :param list ml: each element is a list of indices of points that must be in the same cluster
        :param list cl: each element is a list of indices of points that cannot be in the same cluster
        :param hyperparameters: hyperparameters of distributions for the variates
        :type hyperparameters: hyperparameters or None
        :return: the clustering solution
        :rtype: GibbsCluster
        """

        assert isinstance(X, pd.DataFrame), 'X must be a pandas.DataFrame'
        ml_flat = [idx for link in ml for idx in link]
        cl_flat = [idx for link in cl for idx in link]
        assert not(is_duplicate(ml_flat)), 'Integers in ml are not unique'
        assert (0 <= np.array(ml_flat)).all() and (np.array(ml_flat) < len(X)).all() \
            and all([isinstance(item, int) for item in ml_flat]), 'Elements of ml must be integers in [0,len(X))'
        assert (0 <= np.array(cl_flat)).all() and (np.array(cl_flat) < len(X)).all() \
            and all([isinstance(item, int) for item in cl_flat]), 'Elements of cl must be integers in [0,len(X))'
        # order the columns of X so Gaussian variates come first, then Bernoullis then Poissons
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

        # perform Gibbs sampling
        clust_soln = self._gibbs(X, hyperparameters, ml, cl)

        return clust_soln

    def predict(self, X):
        """
        Predicts clusters for the observations (rows) in X

        :param pd.DataFrame X: data
        :return: the clustering solution
        :rtype: np.array, np.ndarray
        """

        assert self.phi is not None and self.clusters is not None, 'Model has not been fitted'
        # order the columns of X so Gaussian variates come first, then Bernoullis then Poissons
        X, n_cts_data, n_bin_data, n_ord_data = sort_features(X)
        n_cts = len(self.phi[0]['mu'])
        n_bin = len(self.phi[0]['p'])
        n_ord = len(self.phi[0]['lam'])
        assert n_cts+n_bin+n_ord == len(X.columns), 'The features in X do not match the features of the model'
        n = len(X)
        uniq_c = np.unique(self.clusters)  # array of cluster indices
        weights = Counter(self.clusters)  # weights of the clusters
        prob = np.zeros((n, len(uniq_c)))  # initialize probability matrix
        # construct the matrix
        for idx in range(n):
            prob[idx, :] = [weights[clust]*likelihood(X.iloc[idx], self.phi[clust]) for clust in uniq_c]
        # normalize prob
        rowSums = prob.sum(axis=1)[:, None]  # sum probabilities for each observation
        # if all probabilities for an observation are below machine epsilon then it won't be assigned a cluster
        assert 0 not in rowSums, \
            str(list(rowSums).count(0)) + ' observations could not be assigned to a cluster. Increase max_clusters'
        prob = prob / rowSums  # normalize prob
        # pick most probable cluster for each observation
        c = np.array(prob.argmax(axis=1)).reshape(n)
        return c, prob

