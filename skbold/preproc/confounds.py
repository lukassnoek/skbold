"""
The confounds module contains code to handle and account for
confounds in pattern analyses.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ConfoundRegressor(BaseEstimator, TransformerMixin):
    """ Fits a confound onto each feature in X and returns their residuals."""

    def __init__(self, confound, X, cross_validate=True,
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
            (cross_validate=False); we recommend setting this to True to get
            an unbiased estimate.
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
        self.stack_intercept = stack_intercept
        self.weights_ = None

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
        if self.confound.squeeze().ndim == 1 and self.stack_intercept:
            intercept = np.ones(self.confound.shape[0])
            self.confound = np.column_stack((intercept, self.confound))

        confound = self.confound

        # fit_idx = np.in1d(self.X, X).reshape(self.X.shape).sum(axis=1) == self.X.shape[1]
        fit_idx = np.in1d(self.X.sum(axis=1), X.sum(axis=1))  # not guaranteed to work

        c = confound[fit_idx, :]

        # Vectorized implementation estimating weights for all voxels simultaneously
        self.weights_ = np.linalg.pinv(c.T.dot(c)).dot(c.T).dot(X)
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

        #fit_idx = np.in1d(self.X, X).reshape(self.X.shape).sum(axis=1) == self.X.shape[1]
        fit_idx = np.in1d(self.X.sum(axis=1), X.sum(axis=1))  # not guaranteed to work

        c = self.confound[fit_idx]
        X_new = X - c.dot(self.weights_)
        return X_new
