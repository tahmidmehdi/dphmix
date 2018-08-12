# dphmix
# Unsupervised and Semi-supervised Dirichlet Process Heterogeneous Mixtures #

A Python package that implements Dirichlet Process Heterogeneous Mixtures (DPHMs) of exponential family distributions for clustering heterogeneous data without choosing the number of clusters. Inference can be performed with Gibbs sampling [2] or coordinate ascent mean-field variational inference (MFVI) [1]. For semi-supervised learning, Gibbs sampling supports must-link and can't-link constraints [3]. A novel variational inference algorithm was derived to handle must-link constraints.

It currently supports the following distributions:

* Normal: continuous variables are inferred to be Gaussian
* Bernoulli: binary {0,1} variables are inferred to be Bernoullian
* Poisson: non-negative discrete variables are inferred to be Poissonian

## Dependencies ##

* joblib
* numpy
* pandas
* scipy

Tested on Python 3.6.4

## Installation ##

`pip install dphmix`


## Classes ##

### VariationalDPHM ###

`VariationalDPHM(alpha, iterations, max_clusters, tol=1e-3, n_jobs=1, random_state=None)`

Implements MFVI with conjugate priors using the Stick-breaking Process for Dirichlet Process Mixture Models where the independent variates follow Normal, Bernoulli & Poisson distributions. It can also handle must-link constraints.

For MFVI, add 'sb1' & 'sb2' to hyperparameters for variational posterior of v. Variational expectations are contained in dictionaries where keys=['lntau','tau','mu','lnp','lnip','lnlam','lam','lnv','lniv'] where the values of 'mu','tau' & 'lntau' must be lists of numbers with length G, the value of 'lnp' & 'lnip' must be a list of numbers with length B, the value of 'lam' & 'lnlam' must be a list of numbers with length P and the values of 'lnv' & 'lniv' must be numbers.

Parameter | Data type | Description
:---: | :---: | :---
alpha | float>0 | required. The alpha parameter for the Dirichlet Process. Determines how precisely the model should look for clusters. Higher values will create more clusters.
iterations | int>=1 | required. The maximum number of iterations.
max\_clusters | int>=2 | required. The maximum number of clusters the model can create.
tol | float>0 | optional (default: 1e-3). If using MFVI, the algorithm stops when the difference between evidence lower bounds (ELBOs) in 2 consecutive iterations is less than tol.
n\_jobs | int>=1 | optional (default: 1). The number of cores to use.
random\_state | int>=0 or None | optional (default: None). If using MFVI, this determines the initial clusters and ensures reproducibility.

Method | Description
:---: | :---
fit_predict(X[, ml, hyperparameters]) | Fits a model & predicts cluster assignments for each observation in X. ml can be passed to force certain observations to cluster together.
predict(X) | Predicts cluster assignments for each observation in X along with probabilities of belonging to each cluster using the fitted model.
get\_params() | Gets arguments for the model.

`fit_predict(X, ml=[], hyperparameters=None)`

Returns a MFVICluster object.

Parameter | Data type | Description
:---: | :---: | :---
X | pandas.DataFrame | required. The data to cluster. Rows are observations and columns are variables.
ml | list | optional. A list of must-link constraints where a must-link constraint is a list of indices of observations that must cluster together.
hyperparameters | dict or None | optional (there are built-in default hyperparameters). If X has G, B & P continuous, binary & ordinal features, respectively, then hyperparameters are contained in dictionaries where keys=['nu','rho','a','b','gamma','delta','eps','zeta']. The values of 'nu','rho','a' & 'b' must be lists of numbers with length G (all positive except 'nu'), the values of 'gamma' & 'delta' must be lists of positive numbers with length B, the values of 'eps' & 'zeta' must be lists of positive numbers with length P. 'nu', 'rho', 'a' & 'b' are the mean, precision, shape and rate parameters, respectively, of the NormalGamma distributions that generate means and precisions of Gaussian features. 'gamma' & 'delta' are the shape parameters of the Beta distributions that generate probability parameters for Bernoullian features. 'eps' & 'zeta' are the shape and rate parameters, respectively, of the Gamma distributions that generate average rate parameters for Poissonian features. In every list, the values must be in the same order that their corresponding features appear in X.

`predict(X)`

Returns a vector of cluster assignments for observations in X and a probability matrix where rows correspond to observations & columns correspond to clusters.

Parameter | Data type | Description
:---: | :---: | :---
X | pandas.DataFrame | required. The data to cluster.

`get_params()`

Returns a dictionary of arguments for the model

### GibbsDPHM ###

`GibbsDPHM(alpha, iterations, max_clusters, n_jobs=1)`

Implements a Gibbs sampler with conjugate priors using the Chinese Restaurant Process for Dirichlet Process Mixture Models where the independent variates follow Normal, Bernoulli & Poisson distributions. It can handle must-link & can't-link constraints.

Parameters are contained in dictionaries where keys=['mu','tau','p','lam'] where the values of 'mu' & 'tau' are lists of numbers with length G, the value of 'p' is a list of numbers with length B and the value of 'lam' is a list of numbers with length P. 'mu' & 'tau' represent means and precisions, respectively, of Gaussian features, 'p' represents probability parameters of Bernoullian features and 'lam' represents average rate parameters of Poissonian features. In every list, values are in the same order that their corresponding features appear in the data.

Parameter | Data type | Description
:---: | :---: | :---
alpha | float>0 | required. The alpha parameter for the Dirichlet Process. Determines how precisely the model should look for clusters. Higher values will create more clusters.
iterations | int>=1 | required. The maximum number of iterations.
max\_clusters | int>=2 | required. The maximum number of clusters the model can create.
n\_jobs | int>=1 | optional (default: 1). The number of cores to use.

Method | Description
:---: | :---
fit_predict(X[, ml, cl, hyperparameters]) | Fits a model & predicts cluster assignments for each observation in X. ml can be passed to force certain observations to cluster together. cl can be passed to force certain observations to be in different clusters.
predict(X) | Predicts cluster assignments for each observation in X along with probabilities of belonging to each cluster using the fitted model.
get\_params() | Gets arguments for the model.

`fit_predict(X, ml=[], cl=[], hyperparameters=None)`

Returns a GibbsCluster object.

Parameter | Data type | Description
:---: | :---: | :---
X | pandas.DataFrame | required. The data to cluster. Rows are observations and columns are variables.
ml | list | optional. A list of must-link constraints.
cl | list | optional. A list of can't-link constraints where a can't-link constraint is a list of 2 indices corresponding to observations that cannot cluster together.
hyperparameters | dict or None | optional (there are built-in default hyperparameters). Same as the VariationalDPHM.fit_predict

`predict(X)`

Returns a vector of cluster assignments for observations in X and a probability matrix where rows correspond to observations & columns correspond to clusters.

Parameter | Data type | Description
:---: | :---: | :---
X | pandas.DataFrame | required. The data to cluster.

`get_params()`

Returns a dictionary of arguments for the model
### MFVICluster ###

Attribute | Data type | Description
:---: | :---: | :---
c | numpy.array | Cluster assignments of the observations in the data in the order they appear in the data.
variational\_parameters | list | List of parameters where the ith element is a dict of variational parameters for the ith cluster.
moments | list | List of moments where the ith element is a dict of expectations of distribution parameters for the ith cluster.
Ez | numpy.ndarray | matrix where the element in row i, column j is the probability of observation i being in cluster j.
cluster\_map | dict | mapping of cluster indices in the model (0-max\_clusters) to indices in c.
n\_clusters | int>=1 | The number of clusters found
ELBO | float<=0 | The final ELBO

### GibbsCluster ###

Attribute | Data type | Description
:---: | :---: | :---
c | numpy.array | Cluster assignments of the observations in the data in the order they appear in the data.
phi | list | List of parameters where the ith element is a dict of parameters for the ith cluster.
n\_clusters | int>=1 | The number of clusters found

## References: ##

[1] DM Blei & MI Jordan. (2006). Variational Inference for Dirichlet Process Mixtures. Bayesian Analysis, 1(1), 121-144.

[2] RM Neal. (2000). Markov Chain Sampling Methods for Dirichlet Process Mixture Models. Journal of Computational and Graphical Statistics, 9(2), 249-265.

[3] A Vlachos, A Korhonen & Z Ghahramani. (2009). Unsupervised and Constrained Dirichlet Process Mixture Models for Verb Clustering. Proceedings of the EACL 2009 Workshop on GEMS: GEometical Models of Natural Language Semantics, 74-82.

