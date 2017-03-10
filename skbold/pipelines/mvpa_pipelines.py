# Contains some standard pipelines

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import (f_classif, SelectKBest,
                                       SelectPercentile)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def create_ftest_kbest_svm(kernel='linear', k=100, **kwargs):
    """ Creates an svm-pipeline with f-test feature selection.

    Uses SelectKBest from scikit-learn.feature_selection.

    Parameters
    ----------
    kernel : str
        Kernel for SVM (default: 'linear')
    k : int
        How many voxels to select (from the k best)
    **kwargs
        Arbitrary keyword arguments for SVC() initialization.

    Returns
    -------
    ftest_svm : scikit-learn Pipeline object
        Pipeline with f-test feature selection and svm.
    """

    ftest_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('ufs', SelectKBest(f_classif, k=k)),
        ('svm', SVC(kernel=kernel, **kwargs))
    ])
    return ftest_svm


def create_ftest_percentile_svm(kernel='linear', perc=10, **kwargs):
    """ Creates an svm-pipeline with f-test feature selection.

    Uses SelectPercentile from scikit-learn.feature_selection.

    Parameters
    ----------
    kernel : str
        Kernel for SVM (default: 'linear')
    perc : int or float
        Percentage of voxels to select
    **kwargs
        Arbitrary keyword arguments for SVC() initialization.

    Returns
    -------
    ftest_svm : scikit-learn Pipeline object
        Pipeline with f-test feature selection and svm.
    """
    ftest_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('ufs', SelectPercentile(f_classif, percentile=perc)),
        ('svm', SVC(kernel=kernel, **kwargs))
    ])
    return ftest_svm


def create_pca_svm(kernel='linear', n_comp=10, whiten=False, **kwargs):
    """ Creates an svm-pipeline with f-test feature selection.

    Parameters
    ----------
    kernel : str
        Kernel for SVM (default: 'linear')
    n_comp : int
        How many PCA-components to select
    whiten : bool
        Whether to use whitening in PCA
    **kwargs
        Arbitrary keyword arguments for SVC() initialization.

    Returns
    -------
    pca_svm : scikit-learn Pipeline object
        Pipeline with PCA feature extraction and svm.
    """
    pca_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comp, whiten=whiten)),
        ('svm', SVC(kernel=kernel, **kwargs))
    ])
    return pca_svm
