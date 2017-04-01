# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD


"""
The transformer subpackage provides several scikit-learn style transformers
that perform feature selection and/or extraction of multivoxel fMRI patterns.
Most of them are specifically constructed with fMRI data in mind, and thus
often need an Mvp object during initialization to extract necessary metadata.
All comply with the scikit-learn API, using fit() and transform() methods.
"""

from .filters import GenericUnivariateSelect, SelectAboveCutoff
from .selectors import fisher_criterion_score

__all__ = ['GenericUnivariateSelect', 'SelectAboveCutoff',
           'fisher_criterion_score']
