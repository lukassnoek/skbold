# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
.. _ReadTheDocs: http://skbold.readthedocs.io
The utils subpackage contains for extra utilities for machine learning
pipelines on fMRI data. The MvpResults* objects can be used to keep track of
model evaluation metrics across cross-validation folds and keeps track of
feature weights. More information can be found on the homepage of
ReadTheDocs_.
"""


from .mvp_results import MvpResults, MvpResultsClassification, MvpResultsRegression, MvpAverageResults
from .sort_numbered_list import sort_numbered_list
from .decoding_splits import make_counterbalanced_split

__all__ = ['MvpResultsClassification', 'MvpResults',
           'MvpResultsRegression', 'MvpAverageResults',
           'sort_numbered_list', 'make_counterbalanced_split',
           'MvpAverageResults']