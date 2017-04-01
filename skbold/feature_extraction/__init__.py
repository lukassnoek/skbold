# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
This module contains some feature-extraction methods/transformers.
"""

from .transformers import (PCAfilter, ArrayPermuter, PatternAverager,
                           AverageRegionTransformer, RoiIndexer, RowIndexer,
                           ClusterThreshold, SelectFeatureset,
                           IncrementalFeatureCombiner)

__all__ = ['PatternAverager', 'ArrayPermuter', 'AverageRegionTransformer',
           'PCAfilter', 'RoiIndexer', 'RowIndexer', 'ClusterThreshold',
           'SelectFeatureset', 'IncrementalFeatureCombiner']
