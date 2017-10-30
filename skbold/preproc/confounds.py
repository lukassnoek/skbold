"""
The confounds module contains code to handle and account for
confounds in pattern analyses.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ConfoundRegressor(BaseEstimator, TransformerMixin):
    """ Fits a confound onto each feature in X and returns residuals."""

    def __init__(self, confound, X, cross_validate=True,
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

        self.confound = confound
        self.cross_validate = cross_validate
        self.X = X
        self.stack_intercept = stack_intercept
        self.weights_ = None

    def fit(self, X, y=None):

        if self.confound.ndim == 1 and self.stack_intercept:
            intercept = np.ones(self.confound.shape[0])
            self.confound = np.column_stack((intercept, self.confound))

        confound = self.confound
        fit_idx = np.in1d(self.X, X).reshape(self.X.shape).sum(axis=1) == self.X.shape[1]
        confound = confound[fit_idx, :]
        weights = np.zeros((X.shape[1], confound.shape[1]))
        for i in range(X.shape[1]):
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

        if not self.cross_validate:
            self.fit(X)

        fit_idx = np.in1d(self.X, X).reshape(self.X.shape).sum(axis=1) == self.X.shape[1]
        confound = self.confound[fit_idx]
        X_new = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_new[:, i] = X[:, i] - np.squeeze(confound.dot(self.weights_[i, :]))

        return X_new


if __name__ == '__main__':

    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import numpy as np

    X = np.random.normal(0, 1, (100, 500))
    y = np.repeat([0, 1], 50)
    c = np.random.normal(0, 1, 100)
    skf = StratifiedKFold(10)

    pipeline = [('cr', ConfoundRegressor(c, X, cross_validate=False)),
                ('clf', SVC(kernel='linear'))]
    pipeline = Pipeline(pipeline)
    cross_val_score(pipeline, X, y, cv=skf)
