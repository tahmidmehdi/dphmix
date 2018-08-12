"""
Tahmid F. Mehdi
Moses Lab, University of Toronto
Cluster Classes
April 22, 2018

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

import numpy as np


class GibbsCluster:
    def __init__(self, cluster_assignments, parameters):
        """
        Construct a GibbsCluster object

        :param array-like cluster_assignments: list of clusters
        :param list parameters: parameters of each cluster
        """

        self.c = np.array(cluster_assignments)
        self.phi = parameters
        self.n_clusters = len(np.unique(cluster_assignments))

    def __repr__(self):
        return "{} clusters from a Gibbs Sampler".format(self.n_clusters)

    def __str__(self):
        return "{} clusters from a Gibbs Sampler".format(self.n_clusters)


class MFVICluster:
    def __init__(self, cluster_assignments, var_pars, moments, prob_matrix, cluster_mapping, elbo):
        """
        Construct a MFVICluster object

        :param array-like cluster_assignments: list of clusters
        :param list var_pars: variational parameters
        :param list moments: expectations for cluster parameters
        :param np.ndarray prob_matrix: matrix where the element in row i, column j is the probability of observation i
        being in cluster j
        :param dict cluster_mapping: mapping of cluster indices in the model (0-max_clusters) to indices in
        cluster_assignments
        :param float<=0 elbo: the final ELBO
        """

        self.c = np.array(cluster_assignments)
        self.variational_parameters = var_pars
        self.moments = moments
        self.Ez = prob_matrix
        self.cluster_map = cluster_mapping
        self.n_clusters = len(np.unique(cluster_assignments))
        self.ELBO = elbo

    def __repr__(self):
        return "{} clusters from coordinate ascent mean-field variational inference with ELBO={}"\
            .format(self.n_clusters, self.ELBO)

    def __str__(self):
        return "{} clusters from coordinate ascent mean-field variational inference with ELBO={}" \
            .format(self.n_clusters, self.ELBO)