# dphmix
# Unsupervised and Semi-supervised Dirichlet Process Heterogeneous Mixtures #

A Python package that implements Dirichlet Process Heterogeneous Mixtures (DPHMs) of exponential family distributions for clustering heterogeneous data without choosing the number of clusters. Inference can be performed with Gibbs sampling [1] or coordinate ascent mean-field variational inference (MFVI) [2]. For semi-supervised learning, Gibbs sampling supports must-link and can't-link constraints [3]. A novel variational inference algorithm was derived to handle must-link constraints.

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

## Tutorial ##

In this tutorial, we cluster genomic regions bound by the Nkx2-5 transcription factor based on biological features.

1. Download the Nkx2-5 dataset from https://github.com/tahmidmehdi/dphmix/tree/master/data.
2. In Python, import the package and store the data in a dataframe. The rows represent observations and columns represent features:
```python
from dphmix.VariationalDPHM import *
X = pd.read_csv('nkxData.csv', indexcol=0, header=0)
# drop cluster and class assignment columns. These are just results from our paper.
X.drop(['Cluster', 'Class'] , axis=1, inplace=True)
```
3. Set hyperparameters for the model. Descriptions for hyperparameters are shown in the next section.
```python
# prior parameters for 100 Gaussian features
nu = [0]*100
rho = [1]*100
a = [1]*100
b = [1]*100
# prior parameters for 5 Bernoulli features
gamma = [1]*5
delta = [1]*5
# prior parameters for 9 Poisson features
zeta = 1/np.std(X[X.columns[105:]])
eps = zeta*np.mean(X[X.columns[105:]])
# dictionary of hyperparameters
h = dict(nu=nu, rho=rho, a=a, b=b, gamma=gamma, delta=delta, eps=eps, zeta=zeta)
```
4. Initialize a 'VariationalDPHM' model and fit it to X. Information about parameters and outputs is provided in the next section.
```python
model = VariationalDPHM(alpha=1, iterations=1000 , max clusters=100 , tol=10, random_state=42)
clusters = model.fit_predict(X, hyperparameters=h)
```
Features with 'float' data types are Gaussian. Features with 'int' data types that only have values of 0 or 1 are Bernoulli. Features with 'int' that only have non-negative values with at least one integer greater than 1 are Poisson. If your feature has negative integer values, you can add a constant to them to shift them to the non-negative integer space.

## Classes ##

### VariationalDPHM ###

`VariationalDPHM(alpha, iterations, max_clusters, tol=1e-3, n_jobs=1, random_state=None)`

Implements MFVI with conjugate priors using the Stick-breaking Process for Dirichlet Process Mixture Models where the independent variates follow Normal, Bernoulli & Poisson distributions. It can also handle must-link constraints.

Parameter | Data type | Description
:---: | :---: | :---
alpha | float>0 | required. The alpha parameter for the Dirichlet Process. Determines how precisely the model should look for clusters. Higher values will create more clusters.
iterations | int>=1 | required. The maximum number of iterations.
max\_clusters | int>=2 | required. The maximum number of clusters the model can create.
tol | float>0 | optional (default: 1e-3). The algorithm stops when the difference between evidence lower bounds (ELBOs) in 2 consecutive iterations is less than tol.
n\_jobs | int>=1 | optional (default: 1). The number of cores to use.
random\_state | int>=0 or None | optional (default: None). Determines the initial clusters and ensures reproducibility.

Method | Description
:---: | :---
fit\_predict(X[, ml, hyperparameters]) | Fits a DPHM model & predicts cluster assignments for each observation in X. ml can be passed to force certain observations to cluster together.
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

Parameter | Data type | Description
:---: | :---: | :---
alpha | float>0 | required. The alpha parameter for the Dirichlet Process. Determines how precisely the model should look for clusters. Higher values will create more clusters.
iterations | int>=1 | required. The maximum number of iterations.
max\_clusters | int>=2 | required. The maximum number of clusters the model can create.
n\_jobs | int>=1 | optional (default: 1). The number of cores to use.

Method | Description
:---: | :---
fit\_predict(X[, ml, cl, hyperparameters]) | Fits a model & predicts cluster assignments for each observation in X. ml can be passed to force certain observations to cluster together. cl can be passed to force certain observations to be in different clusters.
predict(X) | Predicts cluster assignments for each observation in X along with probabilities of belonging to each cluster using the fitted model.
get\_params() | Gets arguments for the model.

`fit_predict(X, ml=[], cl=[], hyperparameters=None)`

Returns a GibbsCluster object.

Parameter | Data type | Description
:---: | :---: | :---
X | pandas.DataFrame | required. The data to cluster. Rows are observations and columns are variables.
ml | list | optional. A list of must-link constraints.
cl | list | optional. A list of can't-link constraints where a can't-link constraint is a list of 2 indices corresponding to observations that cannot cluster together.
hyperparameters | dict or None | optional (there are built-in default hyperparameters). Same as the VariationalDPHM.fit\_predict hyperparameters

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

## Citation ##

Tahmid F Mehdi, Gurdeep Singh, Jennifer A Mitchell, Alan M Moses, Variational infinite heterogeneous mixture model for semi-supervised clustering of heart enhancers, Bioinformatics, Volume 35, Issue 18, 15 September 2019, Pages 3232–3239, https://doi.org/10.1093/bioinformatics/btz064

## References: ##

[1] RM Neal. (2000). Markov Chain Sampling Methods for Dirichlet Process Mixture Models. Journal of Computational and Graphical Statistics, 9(2), 249-265.

[2] DM Blei & MI Jordan. (2006). Variational Inference for Dirichlet Process Mixtures. Bayesian Analysis, 1(1), 121-144.

[3] A Vlachos, A Korhonen & Z Ghahramani. (2009). Unsupervised and Constrained Dirichlet Process Mixture Models for Verb Clustering. Proceedings of the EACL 2009 Workshop on GEMS: GEometical Models of Natural Language Semantics, 74-82.
