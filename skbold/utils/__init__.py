# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
The utils subpackage contains some extra utilities for machine learning
pipelines on fMRI data. For example, the `CrossvalSplitter` creates
balanced train/test-sets given a (set of) confound(s). Also,
the `load_roi_mask` function allows for loading ROIs from the
Harvard-Oxford (sub)cortical atlas. This function is also
integrated in the `RoiIndexer` transformer.from

Lastly, the `ArrayPermuter`, `RowIndexer`, and `SelectFeatureset`
transformers can be used in, for example. permutation analyses.
"""

from .sort_numbered_list import sort_numbered_list
from .crossval_splitter import CrossvalSplitter
from .parse_roi_labels import parse_roi_labels
from .load_roi_mask import load_roi_mask, print_mask_options
from .misc_transformers import ArrayPermuter, RowIndexer, SelectFeatureset

__all__ = ['sort_numbered_list', 'CrossvalSplitter',
           'parse_roi_labels', 'print_mask_options',
           'ArrayPermuter', 'RowIndexer', 'SelectFeatureset']
