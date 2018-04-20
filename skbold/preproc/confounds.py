"""
The confounds module contains code to handle and account for
confounds in pattern analyses.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ConfoundRegressor(BaseEstimator, TransformerMixin):
    """ Fits a confound onto each feature in X and returns their residuals."""

    def __init__(self, confound, X, cross_validate=True, precise=False,
                 stack_intercept=True):
        """ Regresses out a variable (confound) from each feature in X.

        Parameters
        ----------
        confound : numpy array
            Array of length (n_samples, n_confounds) to regress out of each
            feature; May have multiple columns for multiple confounds.
        X : numpy array
            Array of length (n_samples, n_features), from which the confound
            will be regressed. This is used to determine how the
            confound-models should be cross-validated (which is necessary
            to use in in scikit-learn Pipelines).
        cross_validate : bool
            Whether to cross-validate the confound-parameters (y~confound)
            estimated from the train-set to the test set (cross_validate=True)
            or whether to fit the confound regressor separately on the test-set
            (cross_validate=False). Setting this parameter to True is equivalent
            to "foldwise confound regression" (FwCR) as described in our paper
            (https://www.biorxiv.org/content/early/2018/03/28/290684). Setting
            this parameter to False, however, is NOT equivalent to "whole
            dataset confound regression" (WDCR) as it does not apply confound
            regression to the *full* dataset, but simply refits the confound
            model on the test-set. We recommend setting this parameter to True.
        precise: bool
            Transformer-objects in scikit-learn only allow to pass the data
            (X) and optionally the target (y) to the fit and transform methods.
            However, we need to index the confound accordingly as well. To do so,
            we compare the X during initialization (self.X) with the X passed to
            fit/transform. As such, we can infer which samples are passed to the
            methods and index the confound accordingly. When setting precise to
            True, the arrays are compared feature-wise, which is accurate, but
            relatively slow. When setting precise to False, it will infer the index
            by looking at the sum of all the features, which is less accurate, but much
            faster. For dense data, this should work just fine. Also, to aid the
            accuracy, we remove the features which are constant (0) across samples.
        stack_intercept : bool
            Whether to stack an intercept to the confound (default is True)

        Attributes
        ----------
        weights_ : numpy array
            Array with weights for the confound(s).
        """

        self.confound = confound
        self.cross_validate = cross_validate
        self.X = X
        self.precise = precise
        self.stack_intercept = stack_intercept
        self.weights_ = None
        self.nonzero_X_ = None

    def fit(self, X, y=None):
        """ Fits the confound-regressor to X.

        Parameters
        ----------
        X : numpy array
            An array of shape (n_samples, n_features), which should correspond
            to your train-set only!
        y : None
            Included for compatibility; does nothing.
        """

        if self.stack_intercept:
            icept = np.ones(self.confound.shape[0])
            self.confound = np.c_[icept, self.confound]

        # Find nonzero voxels (i.e., voxels which have not all zero
        # values across samples)
        if self.nonzero_X_ is None:
            self.nonzero_X_ = np.sum(self.X, axis=0) != 0
            self.X = self.X[:, self.nonzero_X_]

        X_nz = X[:, self.nonzero_X_]
        confound = self.confound

        if self.precise:
            tmp = np.in1d(self.X, X_nz).reshape(self.X.shape)
            fit_idx = tmp.sum(axis=1) == self.X.shape[1]
        else:
            fit_idx = np.in1d(self.X.sum(axis=1), X_nz.sum(axis=1))

        confound_fit = confound[fit_idx, :]

        # Vectorized implementation estimating weights for all features
        self.weights_ = np.linalg.lstsq(confound_fit, X_nz, rcond=None)[0]
        return self

    def transform(self, X):
        """ Regresses out confound from X.

        Parameters
        ----------
        X : numpy array
            An array of shape (n_samples, n_features), which should correspond
            to your train-set only!

        Returns
        -------
        X_new : ndarray
            ndarray with confound-regressed features
        """

        if not self.cross_validate:
            self.fit(X)

        X_nz = X[:, self.nonzero_X_]

        if self.precise:
            tmp = np.in1d(self.X, X_nz).reshape(self.X.shape)
            transform_idx = tmp.sum(axis=1) == self.X.shape[1]
        else:
            transform_idx = np.in1d(self.X.sum(axis=1), X_nz.sum(axis=1))

        confound_transform = self.confound[transform_idx]
        X_new = X - confound_transform.dot(self.weights_)
        X_corr = np.zeros_like(X)
        X_corr[:, self.nonzero_X_] = X_new
        return X_corr
