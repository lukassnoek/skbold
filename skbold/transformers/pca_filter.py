# Class to filter out a (set of) noisy PCA component(s).

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division
from sklearn.base import BaseEstimator, TransformerMixin


class PCAfilter(BaseEstimator, TransformerMixin):
    """
    Will implement a way to regress out a specified number of PCA components,
    which are assumed to be noise components.
    """
    pass
