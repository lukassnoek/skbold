"""
The confounds module contains code to handle and account for
confounds in pattern analyses.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ConfoundRegressor(BaseEstimator, TransformerMixin):
    """ Fits a confound onto y and regresses it out X. """

    def __init__(self, confound, fit_idx, cross_validate=True,
                 stack_intercept=True):
        """ Regresses out a variable (confound) from each feature in X.

        Parameters
        ----------
        confound : list or numpy array
            Array of length n_samples to regress out of each feature;
            May have multiple columns for multiple confounds.
        fit_idx : numpy array of int or bool
            On which samples the confound regressor will be fit
            (this is needed to cross-validate on samples != fit_idx)samples
        cross_validate : bool
            Whether to cross-validate the confound-parameters (y~confound)
            on the test set (cross_validate=True) or whether to fit
            the confound regressor separately on the test-set
            (cross_validate=False)
        stack_intercept : bool
            Whether to stack an intercept to the confound.

        Attributes
        ----------
        weights_ : numpy array
            Array with weights for the confound(s).
        """

        confound = np.array(confound)

        if confound.ndim == 1 or stack_intercept:
            intercept = np.ones(confound.shape[0])
            confound = np.column_stack((intercept, confound))

        self.confound = confound
        self.cross_validate = cross_validate
        self.fit_idx = fit_idx
        self.weights_ = None

    def fit(self, X, y=None):

        confound = self.confound[self.fit_idx]
        weights = np.zeros((X.shape[1], confound.shape[1]))

        for i in range(X.shape[1]):
            print('Fitting feature %i' % i)
            b, _, _, _ = np.linalg.lstsq(confound, X[:, i])
            weights[i, :] = b

        self.weights_ = weights

        return self

    def transform(self, X):
        """ Regresses out confound from X.

        Parameters
        ----------
        X : ndarray
            Numeric (float) array of shape = [n_samples, n_features]
        Returns
        -------
        X_new : ndarray
            ndarray with confound-regressed features
        """

        if X.shape[0] == len(self.fit_idx):
            confound = self.confound[self.fit_idx]
            weights = self.weights_
        else:
            tmp_idx = np.ones(self.confound.shape[0], dtype=bool)
            tmp_idx[self.fit_idx] = False
            confound = self.confound[tmp_idx]

            if not self.cross_validate:
                print("Fitting separately on test")
                self.fit_idx = tmp_idx
                self.fit(X)
                weights = self.weights_

        for i in range(X.shape[1]):
            X[:, i] -= np.squeeze(confound.dot(self.weights_[i, :]))

        return X
