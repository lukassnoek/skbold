# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
The pipelines module contains some standard MVPA pipelines using
the scikit-learn style Pipeline objects.
"""

from .mvpa_pipelines import (create_ftest_kbest_svm, create_pca_svm,
                             create_ftest_percentile_svm)


__all__ = ['create_ftest_kbest_svm', 'create_ftest_percentile_svm',
           'create_pca_svm']
