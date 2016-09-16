# Class to wrap univariate-feature selection methods in.
# Selects features based on {ufs_method}.scores_ > cutoff.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

import numpy as np
from sklearn.feature_selection.univariate_selection import _BaseFilter, check_is_fitted
from sklearn.feature_selection import f_classif


class SelectAboveCutoff(_BaseFilter):
    """ Filter: Select features with a score above some cutoff.

    Parameters
    ----------
    cutoff : int/float
        Cutoff for feature-scores to be selected.
    score_func : callable
        Function that takes a 2D array X (samples x features) and returns a score
        reflecting a univariate difference (higher is better).
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