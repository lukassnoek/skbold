# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
This module contains some feature-extraction methods/transformers.
"""

from .transformers import (PCAfilter, PatternAverager,
                           AverageRegionTransformer, 
                           ClusterThreshold)

__all__ = ['PatternAverager', 'AverageRegionTransformer',
           'PCAfilter', 'ClusterThreshold']
