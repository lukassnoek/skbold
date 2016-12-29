# Class to implement sklearn's f_classif function, but with a minimum
# cutoff instead of an absolute or proportional amount of features.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD


# Class to select features based on their mean euclidean distance between
# average class values.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division, absolute_import
from builtins import range
import numpy as np
from itertools import combinations


def fisher_criterion_score(X, y, norm='l1', balance=False):
    """ Calculates fisher score.

    See [1]_ for more info.

    References
    ----------
    [1] P. E. H. R. O. Duda and D. G. Stork. Pattern Classification.
    Wiley-Interscience Publication, 2001.

    Parameters
    ----------
    X : {array-like, sparse matrix}  shape = (n_samples, n_features)
        The set of regressors that will be tested sequentially.
    y : array of shape(n_samples).
        The data matrix
    norm : str
        Whether to use the l1-norm or l2-norm.

    Returns
    -------
    scores_ : array, shape=(n_features,)
        Fisher criterion scores for each feature.
    """

    n_class = np.unique(y).shape[0]
    n_features = X.shape[1]
    av_patterns = np.zeros((n_class, n_features))

    # Calculate mean patterns
    y_unique = np.unique(y)
    for i in range(n_class):
        av_patterns[i, :] = X[y == y_unique[i], :].mean(axis=0)
        av_patterns[np.isnan(av_patterns)] = 0

    # Create difference vectors, z-score standardization, absolute
    comb = list(combinations(range(1, n_class + 1), 2))
    diff_patterns = np.zeros((len(comb), n_features))
    for i, cb in enumerate(comb):
        a, b = av_patterns[cb[0] - 1], av_patterns[cb[1] - 1, :]
        tmp = a - b

        if norm == 'l1':
            diff_patterns[i, :] = np.abs(tmp / (a.std() + b.std()))
        else:
            diff_patterns[i, :] = (tmp ** 2) / (a.std() ** 2 +
                                                b.std() ** 2)
    if balance:
        scores_ = diff_patterns
    else:
        scores_ = np.mean(diff_patterns, axis=0)

    return scores_
