# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
The classifiers subpackage provides two ensemble-type classifiers that
aim at aggregating multivoxel information from multiple local sources in the
brain. They do so by allowing to fit a model on different brain areas, which
predictions are subsequently combined using either a stacked (meta) model
(i.e. the ``RoiStackingClassifier``) or using a voting-strategy (i.e. the
``RoiVotingClassifier``). The structure and API of these classifiers adhere to
the scikit-learn estimator object.
"""

from .roi_stacking_classifier import RoiStackingClassifier
from .roi_voting_classifier import RoiVotingClassifier

__all__ = ['RoiStackingClassifier', 'RoiVotingClassifier']