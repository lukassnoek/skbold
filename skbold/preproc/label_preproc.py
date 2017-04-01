# Classes preprocess labels ('y').

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division, absolute_import
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats as stat


class LabelFactorizer(BaseEstimator, TransformerMixin):
    """ Transforms labels according to a given factorial grouping.

    Factorizes/encodes labels based on part of the string label. For example,
    the label-vector ['A_1', 'A_2', 'B_1', 'B_2'] can be grouped
    based on letter (A/B) or number (1/2).

    Parameters
    ----------
    grouping : List of str
        List with identifiers for condition names as strings

    Attributes
    ----------
    new_labels_ : list
        List with new labels.
    """

    def __init__(self, grouping):

        self.grouping = grouping
        self.new_labels_ = None

    def fit(self, y=None, X=None):
        """ Does nothing, but included to be used in sklearn's Pipeline. """
        return self

    def transform(self, y, X=None):
        """ Transforms label-vector given a grouping.

        Parameters
        ----------
        y : List/ndarray of str
            List of ndarray with strings indicating label-names
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        y_new : ndarray
            array with transformed y-labels
        X_new : ndarray
            array with transformed data of shape = [n_samples, n_features]
            given new factorial grouping/design.

        """
        y_new = np.zeros(len(y))*-1
        self.new_labels_ = np.array(['parsing error!'] * len(y))

        all_idx = np.zeros(len(y))
        for i, g in enumerate(self.grouping):
            idx = np.array([g in label for label in y])
            y_new[idx] = i
            self.new_labels_[idx] = g
            all_idx += idx

        # Index new labels, y, and X with new factorial labels
        all_idx = all_idx.astype(bool)
        y_new = y_new[all_idx]
        self.new_labels_ = self.new_labels_[all_idx]

        if X is not None:
            X_new = X[all_idx, :]
            return y_new, X_new

        return y_new

    def get_new_labels(self):
        """ Returns new labels based on factorization. """
        return self.new_labels_


class MajorityUndersampler(BaseEstimator, TransformerMixin):
    """
    Undersamples the majority-class(es) by selecting random samples.

    Parameters
    ----------
    verbose : bool
        Whether to print downsamples number of samples.
    """

    def __init__(self, verbose=False):
        """ Initializes MajorityUndersampler object. """
        self.verbose = verbose
        self.idx_ = None

    def fit(self, X=None, y=None):
        """ Does nothing, but included for scikit-learn pipelines. """
        return self

    def transform(self, X, y):
        """ Downsamples majority-class(es).

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        X : ndarray
            Transformed array of shape = [n_samples, n_features] given the
            indices calculated during fit().
        """

        if isinstance(y[0], (np.float64, np.float32, np.float16)):
            print('Converting y to integer')
            y = y.astype(int)

        bins = np.bincount(y)
        all_idx = np.zeros(y.size, dtype=bool)

        for i in np.unique(y):

            if bins[i] != np.min(bins):
                y_idx = y == i
                tmp_idx = np.zeros(y_idx.sum(), dtype=bool)
                idx_idx = np.random.choice(np.arange(y_idx.sum()),
                                           np.min(bins), replace=False)
                tmp_idx[idx_idx] = True
                all_idx[y_idx] = tmp_idx
            else:
                all_idx[y == i] = True

        X_ds, y_ds = X[all_idx, :], y[all_idx]

        if self.verbose:
            print('Number of samples (after resampling): %.3f' % y_ds.size)
            print('Resampled class proportion: %.3f\n' % y_ds.mean())

        self.idx_ = all_idx

        return X[all_idx, :], y[all_idx]


class LabelBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self, params):
        """ Initializes LabelBinarizer object. """
        self.params = params
        self.idx_ = None
        self.binarize_params = None

    def fit(self, X=None, y=None):
        """ Does nothing, but included for scikit-learn pipelines. """
        return self

    def transform(self, X, y):
        """ Binarizes y-attribute.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]

        Returns
        -------
        X : ndarray
            Transformed array of shape = [n_samples, n_features] given the
            indices calculated during fit().
        """

        options = ['percentile', 'zscore', 'constant', 'median']
        params = self.params

        if params['type'] == 'percentile':
            y_rank = [stat.percentileofscore(y, a, 'rank') for a in y]
            y_rank = np.array(y_rank)
            idx = (y_rank < params['low']) | (y_rank > params['high'])
            low = stat.scoreatpercentile(y, params['low'])
            high = stat.scoreatpercentile(y, params['high'])
            self.binarize_params = {'type': 'percentile',
                                    'low': low,
                                    'high': high}
            y = (y_rank[idx] > 50).astype(int)

        elif params['type'] == 'zscore':
            y_norm = (y - y.mean()) / y.std()  # just to be sure
            idx = np.abs(y_norm) > params['std']
            self.binarize_params = {'type': params['type'],
                                    'mean': y.mean(),
                                    'std': y.std(),
                                    'n_std': params['std']}
            y = (y_norm[idx] > 0).astype(int)

        elif params['type'] == 'constant':
            y = (y > params['cutoff']).astype(int)
            idx = None
            self.binarize_params = {'type': params['type'],
                                    'cutoff': params['cutoff']}
        elif params['type'] == 'median':  # median-split
            median = np.median(y)
            y = (y > median).astype(int)
            idx = None
            self.binarize_params = {'type': params['type'],
                                    'median': median}
        else:
            msg = 'Unknown type; please choose from: %r' % options
            raise KeyError(msg)

        if idx is not None:
            X = X[idx, :]

        self.idx_ = idx

        return X, y
