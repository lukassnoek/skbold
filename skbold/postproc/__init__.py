# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
The postproc subpackage contains all off skbold's 'postprocessing'
tools. Most prominently, it contains the MvpResults objects
(both MvpResultsClassification and MvpResultsRegression) which can
be used in analyses to keep track of model performance across
iterations/folds (in cross-validation). Additionally, it allows for
keeping track of feature-scores (e.g. f-values from the
univariate feature selection procedure) or model weights
(e.g. SVM-coefficients). These coefficients can kept track of as
raw weights [1]_ or as 'forward-transformed' weights [2]_.

The postproc subpackage additionally contains the function
'extract_roi_info', which allows to calculate the amount of voxels (and
other statistics) per ROI in a single statistical brain map and output a
csv-file.

The cluster_size_threshold function allows you to set voxels to zero which
do not belong to a cluster of a given extent/size. This is NOT a
statistical procedure (like GRF thresholding), but merely a tool for
visualization purposes.

References
----------
.. [1] Stelzer, J., Buschmann, T., Lohmann, G., Margulies, D.S., Trampel,
R., and Turner, R. (2014). Prioritizing spatial accuracy in
high-resolution fMRI data using multivariate feature weight mapping.
Front. Neurosci., http://dx.doi.org/10.3389/fnins.2014.00066.

.. [2] Haufe, S., Meineck, F., Gorger, K., Dahne, S., Haynes, J-D.,
Blankertz, B., and Biessmann, F. et al. (2014). On the interpretation of
weight vectors of linear models in multivariate neuroimaging. Neuroimage,
87, 96-110.
"""

from .extract_roi_info import extract_roi_info
from .mvp_results import MvpResultsClassification, MvpResultsRegression
from .mvp_results import MvpAverageResults
from .cluster_size_threshold import cluster_size_threshold

__all__ = ['extract_roi_info', 'MvpResultsClassification',
           'MvpResultsRegression', 'MvpAverageResults',
           'cluster_size_threshold']
