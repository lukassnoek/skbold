# Class to wrap univariate-feature selection methods in.
# Selects features based on {ufs_method}.scores_ > cutoff.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import division, print_function, absolute_import
from sklearn.feature_selection.univariate_selection import (_BaseFilter,
                                                            check_is_fitted,
                                                            SelectPercentile,
                                                            SelectFwe,
                                                            SelectFpr,
                                                            SelectFdr,
                                                            SelectKBest)
from sklearn.feature_selection import f_classif
import numpy as np


class SelectAboveCutoff(_BaseFilter):
    """ Filter: Select features with a score above some cutoff.

    Parameters
    ----------
    cutoff : int/float
        Cutoff for feature-scores to be selected.
    score_func : callable
        Function that takes a 2D array X (samples x features) and returns a
        score reflecting a univariate difference (higher is better).
    """

    def __init__(self, cutoff, score_func=f_classif):
        super(SelectAboveCutoff, self).__init__(score_func)
        self.cutoff = cutoff

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')

        if self.scores_.ndim > 1:
            # if at least one column is True, select the feature
            idx = np.sum(self.scores_ > self.cutoff, axis=0)
        else:
            idx = self.scores_ > self.cutoff

        return idx


class GenericUnivariateSelect(_BaseFilter):
    """ Univariate feature selector with configurable strategy.

    Updated version from scikit-learn: http://scikit-learn.org/`.

    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues). For modes 'percentile' or 'kbest' it can return
        a single array scores.
    mode : {'percentile', 'k_best', 'fpr', 'fdr', 'fwe', 'cutoff'}
        Feature selection mode.
    param : float or int depending on the feature selection mode
        Parameter of the corresponding mode.

    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Scores of features.
    pvalues_ : array-like, shape=(n_features,)
        p-values of feature scores, None if `score_func` returned scores only.
    """

    _selection_modes = {'percentile': SelectPercentile,
                        'k_best': SelectKBest,
                        'fpr': SelectFpr,
                        'fdr': SelectFdr,
                        'fwe': SelectFwe,
                        'cutoff': SelectAboveCutoff}

    def __init__(self, score_func=f_classif, mode='percentile', param=1e-5):
        super(GenericUnivariateSelect, self).__init__(score_func)
        self.mode = mode
        self.param = param

    def _make_selector(self):
        selector = self._selection_modes[self.mode](score_func=self.score_func)

        # Now perform some acrobatics to set the right named parameter in
        # the selector
        possible_params = selector._get_param_names()
        possible_params.remove('score_func')
        selector.set_params(**{possible_params[0]: self.param})

        return selector

    def _check_params(self, X, y):
        if self.mode not in self._selection_modes:
            raise ValueError("The mode passed should be one of %s, %r,"
                             " (type %s) was passed."
                             % (self._selection_modes.keys(), self.mode,
                                type(self.mode)))

        self._make_selector()._check_params(X, y)

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')

        selector = self._make_selector()
        selector.pvalues_ = self.pvalues_
        selector.scores_ = self.scores_
        return selector._get_support_mask()
