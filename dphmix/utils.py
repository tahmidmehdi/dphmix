"""
Tahmid F. Mehdi
Moses Lab, University of Toronto
Common functions for GibbsDPHM & VariationalDPHM
July 19, 2018


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


def is_duplicate(l):
    """
    Determines if l has any duplicate entries

    :param list l: list to check
    :return: whether l has any duplicate entries
    :rtype: bool
    """

    for i in range(len(l)):
        for j in range(i+1, len(l)):
            if l[i] == l[j]:
                return True
    return False


def sort_features(X):
    """
    Order the columns of X so Gaussian variates come first, then Bernoullis then Poissons

    :param pd.DataFrame X: data
    :return: X with sorted features & the numbers of continuous, binary & non-negative discrete features
    :rtype: pd.DataFrame, int[>=0], int[>=0], int[>=0]
    """

    n_cts = 0
    n_bin = 0
    n_ord = 0
    ordered_cols = []
    # make a dictionary where keys are data types & values are column names whose data have that type
    dtypes = X.columns.to_series().groupby(X.dtypes).groups
    datatypes = {k.name: v for k, v in dtypes.items()}
    # store continuous variates first
    if 'float64' in datatypes:
        n_cts = len(datatypes['float64'])
        ordered_cols.extend(datatypes['float64'])

    if 'int64' in datatypes:
        bernoulli = []
        poisson = []
        for feature in datatypes['int64']:  # iterate through integer columns
            # if all values of data[feature] are binary, store it as a Bernoulli
            binaries = [datum for datum in X[feature] if datum == 0 or datum == 1]
            if len(binaries) == len(X):
                n_bin += 1
                bernoulli.append(feature)
            else:
                # if all values of data[feature] are discrete & non-negative, store it as a Poisson
                ordinals = [datum for datum in X[feature] if datum >= 0]
                if len(ordinals) == len(X):
                    n_ord += 1
                    poisson.append(feature)

        ordered_cols.extend(bernoulli)
        ordered_cols.extend(poisson)

    X = X[ordered_cols]
    X.columns = [str(col) for col in X.columns]
    X.index = [str(i) for i in X.index]
    return X, n_cts, n_bin, n_ord
