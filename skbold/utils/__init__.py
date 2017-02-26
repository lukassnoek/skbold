# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
.. _ReadTheDocs: http://skbold.readthedocs.io
The utils subpackage contains some extra utilities for machine learning
pipelines on fMRI data. Most notably, the CrossvalSplitter class
allows for the construction of counterbalanced splits between
train- and test-sets (e.g. counterbalancing a certain confounding
variable in the train-set and between the train- and test-set).

More information can be found on the homepage of
ReadTheDocs_.

To do:
- extend crossvalsplitter to create 3 groups (train, cv, test)
"""
import skbold
import os.path as op
from .sort_numbered_list import sort_numbered_list
from .crossval_splitter import CrossvalSplitter
from .parse_roi_labels import parse_roi_labels
from .load_roi_mask import load_roi_mask

__all__ = ['sort_numbered_list', 'CrossvalSplitter',
           'parse_roi_labels']
