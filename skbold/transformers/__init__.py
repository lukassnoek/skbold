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


from .pattern_averager import PatternAverager
from .features_to_contrast import FeaturesToContrast
from .cluster_threshold import ClusterThreshold
from .anova_cutoff import AnovaCutoff
from .array_permuter import ArrayPermuter
from .label_factorizer import LabelFactorizer
from .average_region_transformer import AverageRegionTransformer
from .mean_euclidean import MeanEuclidean
from .pca_filter import PCAfilter
from .roi_indexer import RoiIndexer
from .incremental_feature_combiner import IncrementalFeatureCombiner
from .pca_filter import PCAfilter
from .mean_euclidean_balanced import MeanEuclideanBalanced
from .multi_roi_indexer import MultiRoiIndexer
from .multi_pattern_averager import MultiPatternAverager
from .correlation_selector import CorrelationSelector
from .select_featuresets import SelectFeatureset
from .row_indexer import RowIndexer

__all__ = ['PatternAverager', 'FeaturesToContrast', 'ClusterThreshold',
           'AnovaCutoff', 'ArrayPermuter', 'LabelFactorizer',
           'AverageRegionTransformer', 'MeanEuclidean', 'RoiIndexer',
           'IncrementalFeatureCombiner', 'PCAfilter', 'MeanEuclideanBalanced',
           'MultiRoiIndexer', 'MultiPatternAverager', 'CorrelationSelector',
           'SelectFeatureset', 'RowIndexer']
